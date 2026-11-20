"""Benchmark the optional features on the level-2 benchmark clips:
  --reference : identity lock with a sharp frame taken from a DIFFERENT part of the same source video
  --body      : whole-frame defocus deblur, scored on the region below/around the face (hair, clothes)
Writes bench/extras.md. Run after `lucidface bench run` (needs its degraded clips and manifest)."""
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

from lucidface import backends, bench, track
from lucidface.body import BodyDeblur, Restormer
from lucidface.identity import IdentityLock
from lucidface.restore import restore
from lucidface.videoio import Reader

root = sys.argv[1] if len(sys.argv) > 1 else "bench"
clips = json.load(open(os.path.join(root, "manifest.json")))
dev = "cuda"
scorer = bench.Scorer(dev)
be = backends.get("codeformer", dev, fidelity=1.0)
rows = []


def body_psnr(gt, out, lm):
    """PSNR on the torso/hair region (face box excluded), where --body acts and the face model does not."""
    vals = []
    for i, (a, b) in enumerate(zip(Reader(gt), Reader(out))):
        if i % 10 or np.isnan(lm[i, 0, 0]):
            continue
        iod = np.linalg.norm(lm[i, 1] - lm[i, 0]); cx, cy = lm[i].mean(0)
        H, W = a.shape[:2]
        x0, x1 = int(max(0, cx - 4 * iod)), int(min(W, cx + 4 * iod))
        y0, y1 = int(max(0, cy - 3 * iod)), int(min(H, cy + 6 * iod))
        m = np.zeros((H, W), bool); m[y0:y1, x0:x1] = True
        m[int(cy - 2 * iod):int(cy + 2.2 * iod), int(cx - 1.6 * iod):int(cx + 1.6 * iod)] = False
        d = (a.astype(np.float32) - b.astype(np.float32))[m]
        vals.append(10 * np.log10(255 ** 2 / max((d ** 2).mean(), 1e-6)))
    return float(np.mean(vals))


for c in clips:
    blur = os.path.join(root, f"{c['id']}_L2.mp4")
    gt_tr = track.track(c["gt"], device=dev, log=lambda *_: None)
    tr = track.track(blur, device=dev, log=lambda *_: None)
    n = int(tr["frames"])
    # reference photo: a sharp frame from the source, well outside the benchmark window
    src = os.path.join(root, "src", [f for f in os.listdir(os.path.join(root, "src")) if f.startswith(c["id"])][0])
    t_ref = c["start"] + c["dur"] + 8 if c["start"] < 30 else max(1, c["start"] - 25)
    ref = os.path.join(root, f"{c['id']}_ref.png")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t_ref), "-i", src, "-frames:v", "1", "-update", "1", ref], check=True)
    base = os.path.join(root, f"{c['id']}_L2_cf1.0+lock+steady.25.mp4")
    m0 = scorer.clip(c["gt"], base, gt_tr["landmarks"])
    # reference lock
    out = os.path.join(root, f"{c['id']}_L2_ref.mp4")
    t0 = time.time()
    try:
        idl = IdentityLock([ref], backends.variants(be), device=dev)
        restore(blur, tr, out, be, device=dev, steady=0.25, idlock=idl, log=lambda *_: None)
        m1 = scorer.clip(c["gt"], out, gt_tr["landmarks"])
        rows.append(dict(clip=c["id"], feature="reference", fps=n / (time.time() - t0),
                         id_cos_base=m0["id_cos"], id_cos=m1["id_cos"],
                         psnr_eyes_base=m0["psnr_eyes"], psnr_eyes=m1["psnr_eyes"]))
    except ValueError as e:  # no face in the chosen reference frame
        rows.append(dict(clip=c["id"], feature="reference", error=str(e)))
    print(json.dumps(rows[-1]), flush=True)
    # body deblur
    out = os.path.join(root, f"{c['id']}_L2_body.mp4")
    t0 = time.time()
    body = BodyDeblur(Restormer(dev), tr)
    restore(blur, tr, out, be, device=dev, steady=0.25, body=body, log=lambda *_: None)
    fps = n / (time.time() - t0)
    rows.append(dict(clip=c["id"], feature="body", fps=fps, body_psnr_blurry=body_psnr(c["gt"], blur, gt_tr["landmarks"]),
                     body_psnr_face_only=body_psnr(c["gt"], base, gt_tr["landmarks"]),
                     body_psnr=body_psnr(c["gt"], out, gt_tr["landmarks"])))
    print(json.dumps(rows[-1]), flush=True)
    torch.cuda.empty_cache()

with open(os.path.join(root, "extras.json"), "w") as f:
    json.dump(rows, f, indent=1)
