"""Pass 2: restore every tracked face, on the GPU, in batches.

Per frame:
  1. align   : warp the face to the 512 template using the SMOOTHED landmarks from pass 1
  2. restore : run the backend (CodeFormer / GFPGAN / ...)
  3. lock    : consistency lock in aligned space, r' = r + G_s * (a - r)
               Below the cutoff (iris position and colour, gaze, shading) the result equals what the
               camera recorded; above it (lashes, lid edges, iris texture) it keeps the new detail.
               This is what stops the model from moving the gaze or recolouring the eyes.
  4. steady  : temporal blend of the detail layer d = r' - G_s(r') with the previous frame's,
               motion-compensated by optical flow (RAFT) and switched off where the flow is unreliable
               (blinks, fast mouth). Cuts and missing faces reset it.
  5. paste   : face-parsing mask (skin, brows, eyes, nose, lips), feathered, warped back to the frame.
  6. qc      : per-frame numbers for the QC report (how far the model departed from the
               observation, eye fidelity, eye flicker, focus).
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from . import geometry as G
from .videoio import Reader, Writer

# parsing classes kept in the paste mask (facexlib parsenet, 19 classes): skin, brows, eyes, glasses,
# ears, earring, nose, mouth, lips. Neck (14), hair (17), cloth (16), hat (18) and background stay original.
KEEP = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15]
# eye boxes in template (512) space, used by QC
EYES = [(int(x - 34), int(y - 22), 68, 44) for x, y in G.TEMPLATE[:2]]


class Parser:
    def __init__(self, device):
        from facexlib.parsing import init_parsing_model
        self.net = init_parsing_model(model_name="parsenet", device=device)
        self.keep = torch.zeros(19, device=device)
        self.keep[KEEP] = 1

    @torch.no_grad()
    def mask(self, faces_m11: torch.Tensor) -> torch.Tensor:
        """(B,3,512,512) [-1,1] -> soft mask (B,1,512,512) in [0,1]."""
        lab = self.net(faces_m11)[0].argmax(1)  # (B,512,512)
        m = self.keep[lab].unsqueeze(1)
        m = G.gaussian_blur(G.gaussian_blur(m, 11.0), 11.0)
        m = torch.clamp((m - 0.04) / 0.96, 0, 1)
        e = torch.ones_like(m)
        e[..., :12, :] = 0; e[..., -12:, :] = 0; e[..., :, :12] = 0; e[..., :, -12:] = 0
        return m * G.gaussian_blur(e, 6.0)


class Steadier:
    """Motion-compensated temporal smoothing of the restored detail layer."""

    def __init__(self, device, alpha: float = 0.5, tau: float = 0.035, size: int = 256):
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
        self.raft = raft_small(weights=Raft_Small_Weights.DEFAULT).to(device).eval()
        self.alpha, self.tau, self.size = alpha, tau, size
        self.prev_a = None  # previous aligned input (B=1, 3, 512, 512) in [0,1]
        self.prev_d = None  # previous final detail layer

    def reset(self):
        self.prev_a = self.prev_d = None

    @torch.no_grad()
    def _flow(self, a_from, a_to):
        """Flow that warps a_from onto a_to (backward flow on a_to's grid), at 512."""
        s = self.size
        x = F.interpolate(a_to * 2 - 1, size=(s, s), mode="bilinear", align_corners=False)
        y = F.interpolate(a_from * 2 - 1, size=(s, s), mode="bilinear", align_corners=False)
        fl = self.raft(x, y)[-1]  # flow from a_to to a_from: a_to(p) ~ a_from(p + fl)
        return F.interpolate(fl, size=(G.FACE, G.FACE), mode="bilinear", align_corners=False) * (G.FACE / s)

    @staticmethod
    def _warp(img, flow):
        B, _, H, W = img.shape
        ys, xs = torch.meshgrid(torch.arange(H, device=img.device), torch.arange(W, device=img.device),
                                indexing="ij")
        gx = (xs + flow[:, 0]) / (W - 1) * 2 - 1
        gy = (ys + flow[:, 1]) / (H - 1) * 2 - 1
        return F.grid_sample(img, torch.stack([gx, gy], -1), mode="bilinear", padding_mode="border",
                             align_corners=True)

    @torch.no_grad()
    def __call__(self, a: torch.Tensor, d: torch.Tensor):
        """a: aligned input (1,3,512,512) [0,1]; d: detail layer of this frame. Returns blended d."""
        if self.prev_a is None or self.alpha <= 0:
            self.prev_a, self.prev_d = a, d
            return d, None
        fl = self._flow(self.prev_a, a)
        wa = self._warp(self.prev_a, fl)
        err = G.gaussian_blur((wa - a).abs().mean(1, keepdim=True), 3.0)
        conf = torch.exp(-err / self.tau)  # 1 where the previous frame lines up, ~0 where it does not
        wd = self._warp(self.prev_d, fl)
        out = d + self.alpha * conf * (wd - d)
        self.prev_a, self.prev_d = a, out
        return out, float(conf.mean())


def _to_t(frames, device):
    x = torch.from_numpy(np.stack(frames)).to(device)
    return x.permute(0, 3, 1, 2).flip(1).float() / 255.0  # BGR uint8 -> RGB float


def _to_np(x):
    return (x.clamp(0, 1) * 255 + 0.5).byte().flip(1).permute(0, 2, 3, 1).cpu().numpy()


def restore(video: str, tracks: dict, out_path: str, backend, device: str = "cuda",
            lock_rel: float = 0.066, steady: float = 0.25, batch: int = 8, gate: np.ndarray | None = None,
            body=None, idlock=None, crf: int = 10, log=print) -> dict:
    """lock_rel: consistency-lock cutoff as a fraction of the template inter-ocular distance
    (0 disables). steady: temporal blend strength 0..1 (0 disables). gate: bool per frame,
    False = pass through. body: optional whole-frame defocus deblurrer (lucidface.body), applied before
    the face paste. idlock: optional identity.IdentityLock (reference photos)."""
    rd = Reader(video)
    H, W = rd.h, rd.w
    wr = Writer(out_path, W, H, rd.info.fps_str, crf=crf)
    parser = Parser(device)
    stead = Steadier(device, alpha=steady) if steady > 0 else None
    sig = lock_rel * G.TEMPLATE_IOD
    lm = tracks["landmarks"]
    cuts = set(int(c) for c in tracks.get("cuts", []))
    n = len(lm)
    qc = {k: np.full(n, np.nan, np.float32) for k in ("depart", "eye_depart", "eye_flicker",
                                                         "eye_flicker_in", "steady_conf", "id_sim")}
    qc["restored"] = np.zeros(n, bool)
    qc["variant"] = np.full(n, -1, np.int16)
    t0, i0 = time.time(), 0
    prev_final = prev_in = None
    for frames in rd.batches(batch):
        idx = list(range(i0, i0 + len(frames)))
        i0 += len(frames)
        x = _to_t(frames, device)
        if body is not None:
            x = body(x)
        todo = [k for k, i in enumerate(idx) if i < n and not np.isnan(lm[i, 0, 0])
                and (gate is None or gate[i])]
        if todo:
            ms = [G.similarity(lm[idx[k]]) for k in todo]
            a = G.warp_to_face(x[todo], ms)  # (b,3,512,512) [0,1]
            models = idlock.variants if idlock is not None else [("main", backend)]
            pre, post = [], []
            for _, fn in models:
                rk = (fn(a * 2 - 1) + 1) / 2
                pre.append(rk)
                post.append(rk + G.gaussian_blur(a - rk, sig) if sig > 0 else rk)
            if idlock is not None:
                sims = torch.stack([idlock.similarity(p.clamp(0, 1)) for p in post]).cpu().numpy()
                choice = []
                for j, k in enumerate(todo):
                    i = idx[k]
                    if i in cuts or i == 0 or not qc["restored"][i - 1]:
                        idlock.reset()
                    c = idlock.pick(sims[:, j])
                    choice.append(c)
                    qc["id_sim"][i], qc["variant"][i] = sims[c, j], c
                sel = torch.tensor(choice, device=a.device)
                ar = torch.arange(len(todo), device=a.device)
                r_pre = torch.stack(pre)[sel, ar]
                r = torch.stack(post)[sel, ar]
            else:
                r_pre, r = pre[0], post[0]
            # per-frame QC: how far the model departed from what was filmed (before the lock)
            sq = max(sig, 1.0)
            ra = (G.gaussian_blur(r_pre, sq) - G.gaussian_blur(a, sq)).abs().mean(1)
            mask = parser.mask(r * 2 - 1)
            for j, k in enumerate(todo):
                i = idx[k]
                if stead is not None:
                    if i in cuts or i == 0 or not qc["restored"][i - 1]:
                        stead.reset()
                    base = G.gaussian_blur(r[j:j + 1], sig) if sig > 0 else r[j:j + 1]
                    d, conf = stead(a[j:j + 1], r[j:j + 1] - base)
                    rj = (base + d).clamp(0, 1)
                    qc["steady_conf"][i] = conf if conf is not None else np.nan
                else:
                    rj = r[j:j + 1].clamp(0, 1)
                m = mask[j:j + 1]
                qc["depart"][i] = float((ra[j] * m[0, 0]).sum() / m.sum().clamp(min=1)) * 255
                ed, ef, efi = [], [], []
                for (ex, ey, ew, eh) in EYES:
                    ed.append(float(ra[j, ey:ey + eh, ex:ex + ew].mean()) * 255)
                    if prev_final is not None and prev_final[0] == i - 1:
                        ef.append(float((rj - prev_final[1])[..., ey:ey + eh, ex:ex + ew].abs().mean()) * 255)
                        efi.append(float((a[j:j + 1] - prev_in)[..., ey:ey + eh, ex:ex + ew].abs().mean()) * 255)
                qc["eye_depart"][i] = np.mean(ed)
                if ef:
                    qc["eye_flicker"][i], qc["eye_flicker_in"][i] = np.mean(ef), np.mean(efi)
                prev_final, prev_in = (i, rj), a[j:j + 1]
                # paste back over the face region only
                x0, y0, x1, y1 = G.face_box(ms[j], H, W)
                fr = G.warp_to_frame(torch.cat([rj, m], 1), [ms[j]], H, W, box=(x0, y0, x1, y1))
                mf = fr[:, 3:4]
                x[k:k + 1, :, y0:y1, x0:x1] = fr[:, :3] * mf + x[k:k + 1, :, y0:y1, x0:x1] * (1 - mf)
                qc["restored"][i] = True
        elif stead is not None:
            stead.reset()
        for f in _to_np(x):
            wr.write(f)
        if i0 % (batch * 60) < batch:
            el = time.time() - t0
            log(f"restore: {i0}/{n} frames  {i0 / max(el, 1e-6):.1f} fps")
    wr.close()
    log(f"restore: done {i0} frames in {time.time() - t0:.0f}s")
    return qc
