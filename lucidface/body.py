"""Whole-frame defocus deblurring for hair, hands and clothes (opt-in: --body restormer).

Model: Restormer single-image defocus deblurring (Zamir et al., CVPR 2022), trained on real lens
defocus (DPDD). Code: MIT. Needs a clone of https://github.com/swz30/Restormer and
Defocus_Deblurring/pretrained_models/single_image_defocus_deblurring.pth (Google Drive link in that
folder's README); point LUCIDFACE_RESTORMER_DIR at the clone.

It runs only on the region around the person (face box grown 2.2x sideways and 3x downwards), tiled,
in fp16, and under the same consistency lock as the face, so it cannot shift colours or shapes.
It is the slow part of LucidFace (about 0.3-1 s per frame on a 24 GB GPU); the benchmark reports it.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

from . import geometry as G


class Restormer:
    def __init__(self, device="cuda", root: str | None = None, tile: int = 512, overlap: int = 32,
                 half: bool = True):
        root = root or os.environ.get("LUCIDFACE_RESTORMER_DIR", "Restormer")
        # load the architecture file by path: Restormer and CodeFormer both ship a package called
        # `basicsr`, so importing it as a package would clash with the face backend.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "lucidface_restormer_arch", os.path.join(root, "basicsr/models/archs/restormer_arch.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        Net = mod.Restormer
        self.net = Net(inp_channels=3, out_channels=3, dim=48, num_blocks=[4, 6, 6, 8],
                       num_refinement_blocks=4, heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                       bias=False, LayerNorm_type="WithBias", dual_pixel_task=False)
        ck = torch.load(os.path.join(root, "Defocus_Deblurring/pretrained_models/"
                                           "single_image_defocus_deblurring.pth"), map_location="cpu")
        self.net.load_state_dict(ck["params"])
        self.net = self.net.to(device).eval()
        if half:
            self.net = self.net.half()
        self.tile, self.overlap, self.half = tile, overlap, half

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x (1,3,H,W) RGB [0,1] -> deblurred, tiled with overlap and linear blending."""
        _, C, H, W = x.shape
        t, o = min(self.tile, H, W), self.overlap
        t -= t % 8
        stride = t - o
        out = torch.zeros_like(x)
        wsum = torch.zeros_like(x[:, :1])
        ramp = torch.ones(t, device=x.device)
        if o:
            ramp[:o] = torch.linspace(0.05, 1, o, device=x.device)
            ramp[-o:] = torch.linspace(1, 0.05, o, device=x.device)
        win = (ramp[:, None] * ramp[None, :])[None, None]
        ys = list(range(0, max(H - t, 0) + 1, stride)) or [0]
        xs = list(range(0, max(W - t, 0) + 1, stride)) or [0]
        if ys[-1] + t < H:
            ys.append(H - t)
        if xs[-1] + t < W:
            xs.append(W - t)
        for y in ys:
            for x0 in xs:
                p = x[..., y:y + t, x0:x0 + t]
                q = self.net(p.half() if self.half else p).float().clamp(0, 1)
                out[..., y:y + t, x0:x0 + t] += q * win
                wsum[..., y:y + t, x0:x0 + t] += win
        return out / wsum.clamp(min=1e-6)


class BodyDeblur:
    """Applied to each frame before the face paste. Needs the tracks to know where the person is."""

    def __init__(self, model, tracks: dict, lock_rel: float = 0.066, strength: float = 1.0):
        self.model, self.lm = model, tracks["landmarks"]
        self.lock_rel, self.strength = lock_rel, strength
        self.i = 0

    def region(self, i, H, W):
        lm = self.lm[i]
        if np.isnan(lm[0, 0]):
            return None
        iod = float(np.linalg.norm(lm[1] - lm[0]))
        cx, cy = lm.mean(0)
        x0, x1 = int(cx - 2.2 * 2.5 * iod), int(cx + 2.2 * 2.5 * iod)
        y0, y1 = int(cy - 2.5 * iod), int(cy + 3 * 2.5 * iod)
        return max(0, x0) // 8 * 8, max(0, y0) // 8 * 8, min(W, x1), min(H, y1), iod

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        for b in range(B):
            i = self.i + b
            reg = self.region(i, H, W) if i < len(self.lm) else None
            if reg is None:
                continue
            x0, y0, x1, y1, iod = reg
            src = x[b:b + 1, :, y0:y1, x0:x1]
            r = self.model(src)
            sig = self.lock_rel * iod
            if sig > 0:
                r = r + G.gaussian_blur(src - r, sig)
            # feather the region edge into the untouched frame
            h, w = y1 - y0, x1 - x0
            f = min(48, h // 4, w // 4)
            yy = torch.arange(h, device=x.device).float()
            xx = torch.arange(w, device=x.device).float()
            ey = torch.clamp(torch.minimum(yy, h - 1 - yy) / max(f, 1), 0, 1)
            ex = torch.clamp(torch.minimum(xx, w - 1 - xx) / max(f, 1), 0, 1)
            m = (ey[:, None] * ex[None, :])[None, None] * self.strength
            x[b:b + 1, :, y0:y1, x0:x1] = r.clamp(0, 1) * m + src * (1 - m)
        self.i += B
        return x
