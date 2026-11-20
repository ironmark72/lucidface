"""Face restoration backends.

A backend takes a batch of aligned 512x512 faces (B, 3, 512, 512), RGB, float in [-1, 1], and
returns the restored batch in the same format. LucidFace never ships model weights: each backend
loads weights the user downloaded themselves, under that model's own licence (see NOTICE.md).
"""
from __future__ import annotations

import os
import sys

import torch


class Restorer:
    name = "base"
    licence = ""

    def __call__(self, faces: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class Passthrough(Restorer):
    """Returns the input unchanged. Used to test the pipeline without any model."""
    name = "passthrough"

    def __call__(self, faces):
        return faces


class CodeFormer(Restorer):
    """CodeFormer (Zhou et al., NeurIPS 2022). Code and weights: NTU S-Lab License 1.0,
    NON-COMMERCIAL use only. Needs a clone of https://github.com/sczhou/CodeFormer with
    weights/CodeFormer/codeformer.pth; point LUCIDFACE_CODEFORMER_DIR at it.
    fidelity w: 1.0 = closest to the input (recommended), lower = more invented detail."""
    name = "codeformer"
    licence = "S-Lab License 1.0 (non-commercial)"

    def __init__(self, device="cuda", fidelity: float = 1.0, root: str | None = None):
        root = root or os.environ.get("LUCIDFACE_CODEFORMER_DIR", "CodeFormer")
        if not os.path.isdir(root):
            raise FileNotFoundError(f"CodeFormer repo not found at {root!r}; set LUCIDFACE_CODEFORMER_DIR")
        if root not in sys.path:
            sys.path.insert(0, root)
        from basicsr.archs.codeformer_arch import CodeFormer as Net  # from the CodeFormer repo
        self.net = Net(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                       connect_list=["32", "64", "128", "256"]).to(device).eval()
        ck = torch.load(os.path.join(root, "weights/CodeFormer/codeformer.pth"), map_location="cpu")
        self.net.load_state_dict(ck["params_ema"])
        self.w = fidelity

    @torch.no_grad()
    def __call__(self, faces):
        return self.net(faces, w=self.w, adain=True)[0].clamp(-1, 1)


class GFPGAN(Restorer):
    """GFPGAN v1.4 (Wang et al., CVPR 2021). Code: Apache-2.0. Weights: trained on FFHQ
    (CC BY-NC-SA dataset); commercial status of the weights is unclear: check before commercial use.
    pip install gfpgan; weights GFPGANv1.4.pth from the GFPGAN v1.3.0 release.
    In our eye tests it over-whitened eyes ("doll eyes"); the consistency lock reduces this."""
    name = "gfpgan"
    licence = "Apache-2.0 code; weights trained on FFHQ (check)"

    def __init__(self, device="cuda", weights: str | None = None):
        from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
        weights = weights or os.environ.get("LUCIDFACE_GFPGAN_WEIGHTS", "GFPGANv1.4.pth")
        self.net = GFPGANv1Clean(out_size=512, num_style_feat=512, channel_multiplier=2,
                                 decoder_load_path=None, fix_decoder=False, num_mlp=8,
                                 input_is_latent=True, different_w=True, narrow=1,
                                 sft_half=True).to(device).eval()
        ck = torch.load(weights, map_location="cpu")
        self.net.load_state_dict(ck.get("params_ema", ck.get("params", ck)))

    @torch.no_grad()
    def __call__(self, faces):
        return self.net(faces, return_rgb=False)[0].clamp(-1, 1)


class _Fidelity(Restorer):
    """A CodeFormer view at a different fidelity that shares the loaded network (no extra memory)."""

    def __init__(self, base: CodeFormer, w: float):
        self.base, self.w, self.name = base, w, f"codeformer-w{w}"

    @torch.no_grad()
    def __call__(self, faces):
        return self.base.net(faces, w=self.w, adain=True)[0].clamp(-1, 1)


def variants(backend: Restorer, fidelities=(1.0, 0.8, 0.6)) -> list[tuple[str, Restorer]]:
    """Candidates for the identity lock. CodeFormer: several fidelities sharing one network.
    Other backends: just the backend itself (add more backends here to widen the choice)."""
    if isinstance(backend, CodeFormer):
        return [(f"w{w}", _Fidelity(backend, w)) for w in fidelities]
    return [(backend.name, backend)]


def get(name: str, device="cuda", **kw) -> Restorer:
    name = name.lower()
    if name == "codeformer":
        return CodeFormer(device, fidelity=kw.get("fidelity", 1.0))
    if name == "gfpgan":
        return GFPGAN(device, weights=kw.get("weights"))
    if name in ("passthrough", "none"):
        return Passthrough()
    raise ValueError(f"unknown backend {name!r}")
