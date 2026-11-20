"""lucidface command line.

  lucidface run IN.mp4 -o OUT.mov                 everything on one machine
  lucidface track IN.mp4 -o tracks.npz            pass 1 (face track, focus score)
  lucidface crop IN.mp4 -o face.mp4 --box-out box.json   cut the face region (to upload less)
  lucidface restore face.mp4 -o restored.mp4      pass 2 (+ qc.npz next to it)
  lucidface composite IN.mp4 restored.mp4 -o OUT.mov --box box.json --qc qc.npz
  lucidface bench download|run --dir bench/       reproducible benchmark
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np


def _dev(a):
    import torch
    if a.device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA is not available; pass --device cpu (very slow) or run on a GPU machine")
    return a.device


def _backend(a, device):
    from . import backends
    return backends.get(a.backend, device, fidelity=a.fidelity, weights=a.weights)


def _settings(a) -> dict:
    keys = ("backend", "fidelity", "lock", "steady", "gate", "reference", "body", "batch")
    return {k: getattr(a, k) for k in keys if hasattr(a, k)}


def cmd_track(a):
    from . import track
    tr = track.track(a.input, device=_dev(a), stride=a.stride)
    track.save(a.output, tr)
    box = track.suggest_crop(tr)
    print(json.dumps({"tracks": a.output, "frames": int(tr["frames"]), "suggested_box": box,
                      "median_focus": float(np.nanmedian(tr["focus"]))}))


def cmd_crop(a):
    from . import track
    tr = track.load(a.tracks) if a.tracks else track.track(a.input, device=_dev(a), stride=a.stride)
    x, y, w, h = track.suggest_crop(tr, margin=a.margin)
    subprocess.run(["ffmpeg", "-v", "error", "-stats", "-y", "-i", a.input, "-an", "-sn", "-dn",
                    "-vf", f"crop={w}:{h}:{x}:{y},format=yuv420p", "-c:v", "libx264", "-crf", str(a.crf),
                    "-preset", "faster", a.output], check=True)
    with open(a.box_out, "w") as f:
        json.dump({"box": [x, y, w, h], "source": os.path.abspath(a.input)}, f)
    print(json.dumps({"box": [x, y, w, h], "output": a.output}))


def _gate(a, tr):
    from . import qc
    if a.gate in (None, "off", "all"):
        return None
    return qc.focus_gate(tr["focus"], float(tr["fps"]), float(a.gate))


def cmd_restore(a, tr=None):
    from . import backends, qc as Q, track
    from .restore import restore
    device = _dev(a)
    tr = tr or (track.load(a.tracks) if a.tracks else None)
    if tr is None:
        tr = track.track(a.input, device=device)
        track.save(os.path.splitext(a.output)[0] + ".tracks.npz", tr)
    be = _backend(a, device)
    idl = None
    if a.reference:
        from .identity import IdentityLock
        idl = IdentityLock(a.reference, backends.variants(be), device=device)
    body = None
    if a.body:
        from .body import BodyDeblur, Restormer
        body = BodyDeblur(Restormer(device), tr, lock_rel=a.lock)
    t0 = time.time()
    qcd = restore(a.input, tr, a.output, be, device=device, lock_rel=a.lock, steady=a.steady,
                  batch=a.batch, gate=_gate(a, tr), body=body, idlock=idl)
    fl = Q.flags(qcd, tr)
    w = Q.weights(qcd, fl, float(tr["fps"]))
    qpath = a.qc_out or os.path.splitext(a.output)[0] + ".qc.npz"
    np.savez_compressed(qpath, weights=w, **qcd, **{"flag_" + k: v for k, v in fl.items()})
    print(json.dumps({"output": a.output, "qc": qpath, "seconds": round(time.time() - t0, 1),
                      "restored": int(qcd["restored"].sum()), "flagged": int(fl["any"].sum())}))
    return tr, qcd, fl, w


def cmd_composite(a):
    from .composite import composite
    from .videoio import probe
    box = json.load(open(a.box))["box"] if a.box else None
    if box is None:
        i = probe(a.original)
        box = [0, 0, i.width, i.height]
    w = np.load(a.qc)["weights"] if a.qc else None
    rp = probe(a.restored)
    is_crop = (rp.width, rp.height) == (box[2], box[3]) and (box[2], box[3]) != (probe(a.original).width, probe(a.original).height)
    composite(a.original, a.restored, a.output, tuple(box), restored_is_crop=is_crop, weights=w,
              feather=a.feather, crf=a.crf)
    print(json.dumps({"output": a.output}))


def cmd_report(a):
    from . import qc as Q, track
    from .videoio import probe
    tr = track.load(a.tracks)
    d = np.load(a.qc)
    qcd = {k: d[k] for k in d.files if not k.startswith("flag_") and k != "weights"}
    fl = {k[5:]: d[k] for k in d.files if k.startswith("flag_")}
    box = json.load(open(a.box))["box"] if a.box else None
    rp = probe(a.restored)
    box_r = [0, 0, rp.width, rp.height] if box is None or (rp.width, rp.height) == (box[2], box[3]) else box
    Q.report(a.output, qcd, fl, tr, float(tr["fps"]), a.original, a.restored,
             box or box_r, box_r, {"qc": a.qc})
    print(json.dumps({"report": a.output}))


def cmd_run(a):
    """Everything on one machine: track, restore full frames, composite, report."""
    from . import qc as Q, track
    from .composite import composite
    base = os.path.splitext(a.output)[0]
    a.tracks = None
    tr = track.track(a.input, device=_dev(a))
    track.save(base + ".tracks.npz", tr)
    box = track.suggest_crop(tr)
    rest = base + ".restored.mp4"
    a_in, a.output_final = a.input, a.output
    a.output, a.qc_out = rest, base + ".qc.npz"
    tr, qcd, fl, w = cmd_restore(a, tr)
    composite(a_in, rest, a.output_final, tuple(box), restored_is_crop=False, weights=w,
              feather=a.feather, crf=a.crf)
    Q.report(base + ".qc.html", qcd, fl, tr, float(tr["fps"]), a_in, rest, box, box, _settings(a))
    print(json.dumps({"output": a.output_final, "report": base + ".qc.html"}))


def cmd_bench(a):
    from . import bench, track
    from .restore import restore
    os.makedirs(a.dir, exist_ok=True)
    clips = bench.download(a.dir)
    if a.action == "download":
        return
    device = _dev(a)
    from . import backends
    configs = {
        "blurry": None,
        "cf0.7": dict(backend="codeformer", fidelity=0.7, lock=0, steady=0),
        "cf1.0": dict(backend="codeformer", fidelity=1.0, lock=0, steady=0),
        "cf1.0+lock": dict(backend="codeformer", fidelity=1.0, lock=a.lock, steady=0),
        "cf1.0+lock+steady.25": dict(backend="codeformer", fidelity=1.0, lock=a.lock, steady=0.25),
        "cf1.0+lock+steady.5": dict(backend="codeformer", fidelity=1.0, lock=a.lock, steady=0.5),
        "gfpgan+lock+steady.25": dict(backend="gfpgan", fidelity=1.0, lock=a.lock, steady=0.25),
    }
    if a.only:
        configs = {k: v for k, v in configs.items() if k in a.only.split(",")}
    scorer = bench.Scorer(device)
    rows = []
    for c in clips[: a.clips]:
        gt_tr = track.track(c["gt"], device=device)
        iod = float(np.nanmedian(np.linalg.norm(gt_tr["landmarks"][:, 1] - gt_tr["landmarks"][:, 0], axis=1)))
        for lv in [int(x) for x in a.levels.split(",")]:
            blur = os.path.join(a.dir, f"{c['id']}_L{lv}.mp4")
            if not os.path.exists(blur):
                bench.degrade(c["gt"], blur, bench.LEVELS[lv] * iod)
            tr = track.track(blur, device=device)  # realistic: track on the blurry input
            for name, cfg in configs.items():
                out = os.path.join(a.dir, f"{c['id']}_L{lv}_{name}.mp4")
                t0 = time.time()
                if cfg is None:
                    out = blur
                elif not os.path.exists(out):
                    be = backends.get(cfg["backend"], device, fidelity=cfg["fidelity"])
                    restore(blur, tr, out, be, device=device, lock_rel=cfg["lock"], steady=cfg["steady"],
                            batch=a.batch, log=lambda *_: None)
                fps = gt_tr["frames"] / (time.time() - t0) if cfg is not None else float("nan")
                m = scorer.clip(c["gt"], out, gt_tr["landmarks"])
                rows.append(dict(clip=c["id"], level=lv, config=name, iod=round(iod, 1),
                                 radius_px=round(bench.LEVELS[lv] * iod, 2), fps=fps, **m))
                print(json.dumps(rows[-1]))
                bench.write_table(rows, os.path.join(a.dir, "results.csv"), os.path.join(a.dir, "results.md"))


def main(argv=None):
    p = argparse.ArgumentParser(prog="lucidface", description="Restore out-of-focus faces in video, "
                                "locked to what the camera actually filmed.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--device", default="cuda")

    def restore_args(sp):
        sp.add_argument("--backend", default="codeformer", help="codeformer | gfpgan | passthrough")
        sp.add_argument("--fidelity", type=float, default=1.0, help="CodeFormer w (1.0 = most faithful)")
        sp.add_argument("--weights", default=None, help="weights file for backends that take one")
        sp.add_argument("--lock", type=float, default=0.066,
                        help="consistency-lock cutoff as a fraction of inter-ocular distance (0 = off)")
        sp.add_argument("--steady", type=float, default=0.25, help="temporal blend strength 0-1 (0 = off)")
        sp.add_argument("--gate", default=None, help="focus threshold: restore only frames blurrier than "
                        "this (default: restore every frame)")
        sp.add_argument("--reference", nargs="+", default=None, help="sharp photo(s) of the person")
        sp.add_argument("--body", action="store_true", help="also deblur hair/hands/clothes (slow)")
        sp.add_argument("--batch", type=int, default=8)
        sp.add_argument("--feather", type=int, default=48)
        sp.add_argument("--crf", type=int, default=12)

    s = sub.add_parser("track"); s.add_argument("input"); s.add_argument("-o", "--output", required=True)
    s.add_argument("--stride", type=int, default=1); common(s); s.set_defaults(fn=cmd_track)

    s = sub.add_parser("crop"); s.add_argument("input"); s.add_argument("-o", "--output", required=True)
    s.add_argument("--box-out", required=True); s.add_argument("--tracks"); s.add_argument("--stride", type=int, default=5)
    s.add_argument("--margin", type=float, default=1.6); s.add_argument("--crf", type=int, default=11)
    common(s); s.set_defaults(fn=cmd_crop)

    s = sub.add_parser("restore"); s.add_argument("input"); s.add_argument("-o", "--output", required=True)
    s.add_argument("--tracks"); s.add_argument("--qc-out"); restore_args(s); common(s); s.set_defaults(fn=cmd_restore)

    s = sub.add_parser("composite"); s.add_argument("original"); s.add_argument("restored")
    s.add_argument("-o", "--output", required=True); s.add_argument("--box"); s.add_argument("--qc")
    s.add_argument("--feather", type=int, default=48); s.add_argument("--crf", type=int, default=12)
    s.set_defaults(fn=cmd_composite)

    s = sub.add_parser("report"); s.add_argument("original"); s.add_argument("restored")
    s.add_argument("--tracks", required=True); s.add_argument("--qc", required=True); s.add_argument("--box")
    s.add_argument("-o", "--output", required=True); s.set_defaults(fn=cmd_report)

    s = sub.add_parser("run"); s.add_argument("input"); s.add_argument("-o", "--output", required=True)
    restore_args(s); common(s); s.set_defaults(fn=cmd_run)

    s = sub.add_parser("bench"); s.add_argument("action", choices=["download", "run"])
    s.add_argument("--dir", default="bench"); s.add_argument("--levels", default="1,2,3")
    s.add_argument("--clips", type=int, default=5); s.add_argument("--only")
    s.add_argument("--lock", type=float, default=0.066); s.add_argument("--steady", type=float, default=0.25)
    s.add_argument("--batch", type=int, default=8); common(s); s.set_defaults(fn=cmd_bench)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
