"""Head-to-head: LucidFace vs other face-restoration tools on the same benchmark clips.

Every competitor gets the same degraded clips (bench/<clip>_L<level>.mp4). Tools that only work on a
512x512 face video (PGTFormer, SVFR) get a square crop around the face, resized to 512, and their
output is resized back and pasted into the full frame with a feathered edge, exactly like LucidFace's
own composite. Everything is scored by lucidface.bench.Scorer against the sharp ground truth.

Competitors (each needs its own repo/env; see docs/BENCHMARK.md "Head-to-head"):
  keep      : KEEP (S-Lab licence, non-commercial)       env KEEP_DIR,  KEEP_PY
  pgtformer : PGTFormer (non-commercial BSD-style)       env PGT_DIR,   PGT_PY
  svfr      : SVFR (code MIT, weights non-commercial)    env SVFR_DIR,  SVFR_PY
  topaz     : Topaz Video AI, Iris model, exported by hand at 1x to bench/topaz/<clip>_L<level>.mp4
              (Windows/Mac only; trial exports are watermarked and cannot be scored)

usage: python scripts/headtohead.py BENCH_DIR --level 2 --tools keep,pgtformer,svfr,topaz
Writes BENCH_DIR/h2h.csv and BENCH_DIR/h2h.md.
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import time

import cv2
import numpy as np

from lucidface import bench, track
from lucidface.videoio import Reader, Writer, probe


def square_box(tr, W, H, scale=3.4):
    """One fixed square around the face for the whole clip (median position, size from eye distance)."""
    lm = tr["landmarks"]
    ok = ~np.isnan(lm[:, 0, 0])
    c = np.median(lm[ok].mean(1), axis=0)
    iod = float(np.median(np.linalg.norm(lm[ok, 1] - lm[ok, 0], axis=1)))
    s = int(min(scale * iod, W, H)) // 2 * 2
    x = int(np.clip(c[0] - s / 2, 0, W - s)); y = int(np.clip(c[1] - s / 2 + 0.15 * s, 0, H - s))
    return x, y, s


def crop512(src, dst, box):
    x, y, s = box
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-vf",
                    f"crop={s}:{s}:{x}:{y},scale=512:512:flags=lanczos", "-c:v", "libx264", "-crf", "8",
                    "-pix_fmt", "yuv420p", dst], check=True)


def paste_back(full, face512, out, box, feather=40):
    """Resize the tool's 512 face video back to the box and paste it into the full frame."""
    x, y, s = box
    rd_full, rd_face = Reader(full), Reader(face512)
    yy, xx = np.mgrid[0:s, 0:s]
    d = np.minimum(np.minimum(xx, s - 1 - xx), np.minimum(yy, s - 1 - yy)).astype(np.float32)
    m = np.clip(d / feather, 0, 1)[..., None]
    w = Writer(out, rd_full.w, rd_full.h, rd_full.info.fps_str, crf=10)
    n = 0
    for f, g in zip(rd_full, rd_face):
        f = f.copy()
        g = cv2.resize(g, (s, s), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
        f[y:y + s, x:x + s] = (g * m + f[y:y + s, x:x + s] * (1 - m) + 0.5).astype(np.uint8)
        w.write(f)
        n += 1
    w.close()
    return n


def newest_video(folder):
    vids = [p for p in glob.glob(os.path.join(folder, "**", "*.mp4"), recursive=True)]
    return max(vids, key=os.path.getmtime) if vids else None


def run_keep(blur, out, tmp):
    d, py = os.environ["KEEP_DIR"], os.environ.get("KEEP_PY", "python")
    subprocess.run([py, "inference_keep.py", "-i", blur, "-o", tmp, "--save_video", "-s", "1"], cwd=d, check=True)
    v = newest_video(tmp)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", v, "-c:v", "libx264", "-crf", "8", "-pix_fmt", "yuv420p", out],
                   check=True)


def run_pgtformer(blur, out, tmp, box):
    d, py = os.environ["PGT_DIR"], os.environ.get("PGT_PY", "python")
    face = os.path.join(tmp, "in512.mp4"); res = os.path.join(tmp, "out512.mp4")
    crop512(blur, face, box)
    subprocess.run([py, "inference.py", f"--input_video={face}", f"--output_video={res}"], cwd=d, check=True)
    paste_back(blur, res, out, box)


def run_svfr(blur, out, tmp, box):
    d, py = os.environ["SVFR_DIR"], os.environ.get("SVFR_PY", "python")
    face = os.path.join(tmp, "in512.mp4")
    crop512(blur, face, box)
    subprocess.run([py, "infer.py", "--config", "config/infer.yaml", "--task_ids", "0", "--input_path", face,
                    "--output_dir", tmp], cwd=d, check=True)
    paste_back(blur, newest_video(tmp), out, box)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench"); ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--tools", default="keep,pgtformer,svfr,topaz")
    ap.add_argument("--lucidface", default="cf1.0+lock+steady.25", help="LucidFace config already produced by bench run")
    a = ap.parse_args()
    clips = json.load(open(os.path.join(a.bench, "manifest.json")))
    scorer = bench.Scorer("cuda")
    rows = []
    for c in clips:
        blur = os.path.join(a.bench, f"{c['id']}_L{a.level}.mp4")
        if not os.path.exists(blur):
            print(f"skip {c['id']}: {blur} missing (run `lucidface bench run` first)"); continue
        gt_tr = track.track(c["gt"], log=lambda *_: None)
        tr = track.track(blur, log=lambda *_: None)
        info = probe(blur)
        box = square_box(tr, info.width, info.height)
        n = int(tr["frames"])
        entries = [("blurry", blur, None), ("LucidFace", os.path.join(a.bench, f"{c['id']}_L{a.level}_{a.lucidface}.mp4"), None)]
        for tool in a.tools.split(","):
            out = os.path.join(a.bench, f"{c['id']}_L{a.level}_h2h_{tool}.mp4")
            if tool == "topaz":
                cand = os.path.join(a.bench, "topaz", f"{c['id']}_L{a.level}.mp4")
                entries.append(("Topaz Iris", cand if os.path.exists(cand) else None, None)); continue
            if not os.path.exists(out):
                tmp = os.path.join(a.bench, "_tmp", f"{c['id']}_{tool}"); os.makedirs(tmp, exist_ok=True)
                t0 = time.time()
                try:
                    {"keep": lambda: run_keep(blur, out, tmp),
                     "pgtformer": lambda: run_pgtformer(blur, out, tmp, box),
                     "svfr": lambda: run_svfr(blur, out, tmp, box)}[tool]()
                    with open(out + ".fps", "w") as f:
                        f.write(str(n / (time.time() - t0)))
                except Exception as e:  # a competitor failing must not stop the others
                    print(f"{tool} failed on {c['id']}: {e}", flush=True); out = None
            entries.append((tool, out, None))
        for name, path, _ in entries:
            if not path or not os.path.exists(path):
                rows.append(dict(clip=c["id"], level=a.level, tool=name, status="missing")); continue
            if probe(path).frames != probe(c["gt"]).frames:
                rows.append(dict(clip=c["id"], level=a.level, tool=name, status=f"frame count {probe(path).frames}")); continue
            m = scorer.clip(c["gt"], path, gt_tr["landmarks"])
            fps = open(path + ".fps").read().strip() if os.path.exists(path + ".fps") else ""
            rows.append(dict(clip=c["id"], level=a.level, tool=name, status="ok", fps=fps, **m))
            print(json.dumps(rows[-1]), flush=True)
        with open(os.path.join(a.bench, "h2h.csv"), "w", newline="") as f:
            keys = sorted({k for r in rows for k in r}, key=lambda k: ["clip", "level", "tool", "status"].index(k) if k in ["clip", "level", "tool", "status"] else 9)
            w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
    # summary
    ok = [r for r in rows if r.get("status") == "ok"]
    tools = list(dict.fromkeys(r["tool"] for r in rows))
    lines = ["| tool | clips scored | LPIPS face | identity | eye flicker vs GT | PSNR eyes |", "|---|---|---|---|---|---|"]
    for t in tools:
        rs = [r for r in ok if r["tool"] == t]
        if not rs:
            lines.append(f"| {t} | 0 | - | - | - | - |"); continue
        mean = lambda k: np.mean([float(r[k]) for r in rs])
        lines.append(f"| {t} | {len(rs)} | {mean('lpips_face'):.3f} | {mean('id_cos'):.3f} | {mean('t_err_eyes'):.2f} | {mean('psnr_eyes'):.2f} |")
    open(os.path.join(a.bench, "h2h.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
