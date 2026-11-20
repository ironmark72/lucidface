"""Reproducible benchmark: sharp public-domain interviews -> synthetic lens defocus -> restore ->
score against the sharp ground truth.

Defocus model: a disk point-spread function (what an out-of-focus lens does) with radius proportional
to the inter-ocular distance (so "level 2" means the same thing on a close-up and a wide shot), plus
mild sensor noise and an H.264 re-encode like a real camera file.

Metrics (all in the GT's aligned 512 face space, inside the GT face-parsing mask):
  psnr_face, lpips_face  : fidelity / perceptual distance of the whole face
  psnr_eyes              : the same on the two eye boxes only
  id_cos                 : identity similarity of output vs GT (facenet)
  t_err_eyes             : temporal error, |(out_t - out_t-1) - (gt_t - gt_t-1)| on the eyes;
                           measures flicker that the real footage does not have
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import urllib.request

import cv2
import numpy as np
import torch

from . import geometry as G

# All public domain (US federal government works). Credits are shown in docs/BENCHMARK.md.
CLIPS = [
    dict(id="potus2015", start=20, dur=20, credit="The White House (PD-USGov-POTUS)",
         url="https://upload.wikimedia.org/wikipedia/commons/8/85/2015-02-07_President_Obama%27s_Weekly_Address.webm"),
    dict(id="nasa_murthy", start=10, dur=20, credit="NASA Goddard SVS 14623 (public domain)",
         url="https://svs.gsfc.nasa.gov/vis/a010000/a014600/a014623/EclipseSafety_VivekMurthy_H264.mp4"),
    dict(id="dvids_marceaux", start=4, dur=20, credit="DVIDS (public domain)",
         url="https://upload.wikimedia.org/wikipedia/commons/9/98/Faces_of_Keen_Sword_-_Tech_Sgt_Sasha_Marceaux_%281027106%29.webm"),
    dict(id="dvids_goto", start=10, dur=20, credit="DVIDS (public domain)",
         url="https://upload.wikimedia.org/wikipedia/commons/f/fb/Balikatan_25-_Humanitarian_Assistance_Disaster_Response_exercise_interview_%28960624%29.webm"),
    dict(id="cdc_benjamin", start=10, dur=20, credit="CDC (public domain)",
         url="https://upload.wikimedia.org/wikipedia/commons/f/f5/When_Smoking_Affects_Your_Family%2C_It%E2%80%99s_Personal_%E2%80%93_Former_SG_Regina_Benjamin.webm"),
]
LEVELS = {1: 0.08, 2: 0.12, 3: 0.16}  # disk radius as a fraction of inter-ocular distance


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def download(root: str, log=print) -> list[dict]:
    """Fetch every clip, record its SHA-256, cut the benchmark segment to a lossless-ish mp4."""
    os.makedirs(os.path.join(root, "src"), exist_ok=True)
    out = []
    for c in CLIPS:
        ext = os.path.splitext(c["url"].split("?")[0])[1]
        src = os.path.join(root, "src", c["id"] + ext)
        if not os.path.exists(src):
            log(f"bench: downloading {c['id']}")
            req = urllib.request.Request(c["url"], headers={"User-Agent": "lucidface-bench/0.1"})
            with urllib.request.urlopen(req) as r, open(src, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
        gt = os.path.join(root, f"{c['id']}_gt.mp4")
        if not os.path.exists(gt):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(c["start"]), "-i", src, "-t", str(c["dur"]),
                            "-an", "-c:v", "libx264", "-crf", "6", "-preset", "slow", "-pix_fmt", "yuv420p", gt],
                           check=True)
        out.append({**c, "sha256": sha256(src), "gt": gt})
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def disk_kernel(r: float) -> np.ndarray:
    """Anti-aliased disk PSF of radius r pixels."""
    R = int(np.ceil(r)) + 1
    ss = 4  # supersample the edge
    yy, xx = np.mgrid[-R:R + 1:1 / ss, -R:R + 1:1 / ss]
    d = (xx ** 2 + yy ** 2 <= r ** 2).astype(np.float32)
    k = d.reshape(2 * R + 1, ss, 2 * R + 1, ss).mean((1, 3)) if False else cv2.resize(
        d, (2 * R + 1, 2 * R + 1), interpolation=cv2.INTER_AREA)
    return k / k.sum()


def degrade(gt: str, out: str, radius_px: float, noise: float = 1.5, crf: int = 20, seed: int = 0):
    from .videoio import Reader, Writer
    rd = Reader(gt)
    k = disk_kernel(radius_px)
    rng = np.random.default_rng(seed)
    wr = Writer(out, rd.w, rd.h, rd.info.fps_str, crf=crf)
    for f in rd:
        b = cv2.filter2D(f.astype(np.float32), -1, k, borderType=cv2.BORDER_REFLECT)
        b += rng.normal(0, noise, b.shape).astype(np.float32)
        wr.write(np.clip(b + 0.5, 0, 255).astype(np.uint8))
    wr.close()


class Scorer:
    def __init__(self, device="cuda"):
        import lpips
        from .identity import Embedder
        from .restore import Parser
        self.lp = lpips.LPIPS(net="alex").to(device).eval()
        self.parser = Parser(device)
        self.emb = Embedder(device)
        self.device = device

    @torch.no_grad()
    def clip(self, gt_video: str, out_video: str, gt_landmarks: np.ndarray, batch: int = 8) -> dict:
        from .restore import EYES, _to_t
        from .videoio import Reader
        a, b = Reader(gt_video), Reader(out_video)
        acc = {k: [] for k in ("psnr_face", "lpips_face", "psnr_eyes", "id_cos", "t_err_eyes")}
        prev = None
        i = 0
        for fa, fb in zip(a.batches(batch), b.batches(batch)):
            idx = [i + k for k in range(len(fa)) if not np.isnan(gt_landmarks[i + k, 0, 0])]
            ks = [j - i for j in idx]
            i += len(fa)
            if not idx:
                prev = None
                continue
            ms = [G.similarity(gt_landmarks[j]) for j in idx]
            ga = G.warp_to_face(_to_t([fa[k] for k in ks], self.device), ms)
            ob = G.warp_to_face(_to_t([fb[k] for k in ks], self.device), ms)
            m = (self.parser.mask(ga * 2 - 1) > 0.5).float()
            mse = (((ga - ob) ** 2).mean(1, keepdim=True) * m).sum((1, 2, 3)) / m.sum((1, 2, 3)).clamp(min=1)
            acc["psnr_face"] += (10 * torch.log10(1 / mse.clamp(min=1e-10))).tolist()
            acc["lpips_face"] += self.lp(ga * m * 2 - 1, ob * m * 2 - 1).flatten().tolist()
            eg = torch.cat([ga[..., y:y + h, x:x + w] for x, y, w, h in EYES], -1)
            eo = torch.cat([ob[..., y:y + h, x:x + w] for x, y, w, h in EYES], -1)
            acc["psnr_eyes"] += (10 * torch.log10(1 / ((eg - eo) ** 2).mean((1, 2, 3)).clamp(min=1e-10))).tolist()
            acc["id_cos"] += (self.emb(ga) * self.emb(ob)).sum(1).tolist()
            for j in range(len(idx)):
                if prev is not None and prev[0] == idx[j] - 1:
                    acc["t_err_eyes"].append(float(((eo[j] - prev[2]) - (eg[j] - prev[1])).abs().mean()) * 255)
                prev = (idx[j], eg[j], eo[j])
        return {k: float(np.mean(v)) if v else float("nan") for k, v in acc.items()}


def write_table(rows: list[dict], path_csv: str, path_md: str):
    keys = list(rows[0].keys())
    with open(path_csv, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    # mean over clips per (level, config)
    agg = {}
    for r in rows:
        agg.setdefault((r["level"], r["config"]), []).append(r)
    metrics = ["psnr_face", "lpips_face", "psnr_eyes", "id_cos", "t_err_eyes", "fps"]
    lines = ["| level | config | " + " | ".join(metrics) + " |", "|" + "---|" * (len(metrics) + 2)]
    for (lv, cfg), rs in sorted(agg.items()):
        vals = [np.nanmean([float(r[m]) for r in rs]) for m in metrics]
        lines.append(f"| {lv} | {cfg} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    with open(path_md, "w") as f:
        f.write("\n".join(lines) + "\n")
