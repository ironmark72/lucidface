"""Quality control: which frames to restore, which restorations to trust, and a report a human can
check in minutes instead of watching the whole video.

  focus gate : restore only shots where the face is actually out of focus (per-frame focus score from
               pass 1, median-smoothed, with hysteresis and a minimum shot length)
  flags      : per restored frame, any of
                 depart   - the model's output, re-blurred, is far from what was filmed (invention)
                 shimmer  - eye flicker much higher than the original's own eye motion
                 identity - face drifted from the reference photo (only with --reference)
                 tracking - raw landmarks disagree with the smoothed track (unreliable alignment)
  fallback   : flagged frames fade back to the original (smooth ramp, no hard switch)
  report     : self-contained HTML with timelines, flagged ranges as timestamps, and thumbnails
"""
from __future__ import annotations

import base64
import html
import subprocess

import numpy as np


def _median(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    pad = np.pad(x, (w // 2, w - 1 - w // 2), mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(pad, w), axis=1)


def focus_gate(focus: np.ndarray, fps: float, thr: float, hyst: float = 0.05,
               min_len_s: float = 1.0) -> np.ndarray:
    """True where the face is blurry enough to restore. thr: focus score below which a face counts
    as out of focus (calibrated by `lucidface bench`; see docs/METHOD.md)."""
    f = _median(np.nan_to_num(focus, nan=np.inf), max(1, int(fps)))
    on = np.zeros(len(f), bool)
    state = False
    for i, v in enumerate(f):
        if state and v > thr + hyst:
            state = False
        elif not state and v < thr - hyst:
            state = True
        on[i] = state
    # drop segments shorter than min_len (both on and off) to avoid flicker between modes
    m = int(min_len_s * fps)
    i = 0
    while i < len(on):
        j = i
        while j < len(on) and on[j] == on[i]:
            j += 1
        if j - i < m and i > 0:
            on[i:j] = on[i - 1]
        i = j
    return on


def flags(qc: dict, tracks: dict, depart_thr: float = 7.0, shimmer_ratio: float = 3.0,
          id_thr: float = 0.45, track_rel: float = 0.25) -> dict:
    n = len(qc["restored"])
    rest = qc["restored"]
    lm, raw = tracks["landmarks"][:n], tracks["raw"][:n]
    iod = np.linalg.norm(lm[:, 1] - lm[:, 0], axis=1)
    dev = np.nanmax(np.linalg.norm(raw - lm, axis=2), axis=1) / np.maximum(iod, 1)
    f = {
        "depart": rest & (np.nan_to_num(qc["depart"]) > depart_thr),
        "shimmer": rest & (np.nan_to_num(qc["eye_flicker"]) >
                           shimmer_ratio * np.maximum(np.nan_to_num(qc["eye_flicker_in"]), 0.5)),
        "identity": rest & ~np.isnan(qc["id_sim"]) & (np.nan_to_num(qc["id_sim"], nan=1) < id_thr),
        "tracking": rest & (np.nan_to_num(dev) > track_rel),
    }
    # a single odd frame inside a good run is noise; require 2 of any 3 consecutive frames
    for k, v in f.items():
        c = np.convolve(v.astype(int), [1, 1, 1], mode="same")
        f[k] = v & (c >= 2)
    f["any"] = np.any(np.stack([f[k] for k in ("depart", "shimmer", "identity", "tracking")]), 0)
    return f


def weights(qc: dict, fl: dict, fps: float, ramp_s: float = 0.25) -> np.ndarray:
    """Per-frame blend weight of the restoration (1 = restored, 0 = original), with smooth ramps."""
    w = (qc["restored"] & ~fl["any"]).astype(np.float32)
    r = max(1, int(ramp_s * fps))
    # widen the zero regions by the ramp, then ramp linearly so a flagged frame is fully original
    z = np.convolve((w == 0).astype(np.float32), np.ones(2 * r + 1), mode="same") > 0
    k = np.ones(2 * r + 1) / (2 * r + 1)
    keep = np.pad((~z).astype(np.float32), r, mode="edge")  # edge padding: no dip at start/end
    return np.clip(np.convolve(keep, k, mode="valid"), 0, 1).astype(np.float32)


def ranges(mask: np.ndarray) -> list[tuple[int, int]]:
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def _tc(i: int, fps: float) -> str:
    s = i / fps
    return f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{s % 60:06.3f}"


def _thumb(video: str, t: float, box) -> str:
    x, y, w, h = box
    jpg = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", video, "-frames:v", "1",
                          "-vf", f"crop={w}:{h}:{x}:{y},scale=320:-2", "-f", "image2", "-c:v", "mjpeg",
                          "-q:v", "4", "-"], capture_output=True).stdout
    return "data:image/jpeg;base64," + base64.b64encode(jpg).decode()


def _spark(y: np.ndarray, color: str, h: int = 60, w: int = 900, ymax: float | None = None) -> str:
    y = np.nan_to_num(y.astype(np.float64))
    if len(y) > w:  # downsample by max so spikes stay visible
        k = int(np.ceil(len(y) / w))
        y = np.pad(y, (0, k * w - len(y)), constant_values=0).reshape(w, k).max(1)
    ymax = ymax or (np.percentile(y, 99.5) * 1.2 + 1e-6)
    pts = " ".join(f"{i * w / max(len(y) - 1, 1):.1f},{h - min(v / ymax, 1) * h:.1f}" for i, v in enumerate(y))
    return f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none"><polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1"/></svg>'


def report(path: str, qc: dict, fl: dict, tracks: dict, fps: float, original: str, restored: str,
           box_orig, box_rest, settings: dict, max_thumbs: int = 12):
    n = len(qc["restored"])
    rs = ranges(fl["any"])
    rows = []
    for a, b in rs[:200]:
        why = ", ".join(k for k in ("depart", "shimmer", "identity", "tracking") if fl[k][a:b].any())
        rows.append(f"<tr><td>{_tc(a, fps)}</td><td>{_tc(b, fps)}</td><td>{b - a}</td><td>{why}</td></tr>")
    thumbs = []
    picks = rs[:max_thumbs] if rs else []
    for a, b in picks:
        t = (a + b) / 2 / fps
        thumbs.append(f'<figure><img src="{_thumb(original, t, box_orig)}"><img src="{_thumb(restored, t, box_rest)}">'
                      f"<figcaption>{_tc((a + b) // 2, fps)}: original | restored (shown at full strength)</figcaption></figure>")
    restored_n = int(qc["restored"].sum())
    stat = lambda k: f"{np.nanmedian(qc[k]):.2f}" if np.isfinite(qc[k]).any() else "n/a"
    s_rows = "".join(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>" for k, v in settings.items())
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>LucidFace QC</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{{--bg:#fafaf8;--fg:#1c1c1a;--mut:#6b6b66;--line:#e2e1dc;--a:#2f6fd6;--b:#d0542f;--c:#2c9a5a;--d:#8a5cc7}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151514;--fg:#ecebe6;--mut:#9a9993;--line:#2c2b29}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px;max-width:980px;margin:auto}}
h1{{font-size:22px}} h2{{font-size:16px;margin-top:28px}} .mut{{color:var(--mut)}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}} td,th{{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left}}
svg{{width:100%;height:60px;display:block;border-bottom:1px solid var(--line)}}
.kpi{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}} .kpi div{{border:1px solid var(--line);border-radius:8px;padding:10px}}
.kpi b{{font-size:20px;display:block}} figure{{margin:0 0 16px}} figure img{{width:49%;margin-right:1%}} figcaption{{color:var(--mut);font-size:13px}}
</style></head><body>
<h1>LucidFace QC report</h1>
<p class="mut">{html.escape(original)} &middot; {n} frames at {fps:g} fps</p>
<div class="kpi">
<div><b>{restored_n}</b>frames restored ({100 * restored_n / max(n, 1):.1f}%)</div>
<div><b>{int(fl['any'].sum())}</b>frames flagged, fell back to original</div>
<div><b>{stat('eye_depart')}</b>median eye departure (lower = truer)</div>
<div><b>{stat('id_sim')}</b>median identity similarity</div>
</div>
<h2>Timelines</h2>
<p class="mut">Departure from what was filmed</p>{_spark(qc['depart'], 'var(--b)')}
<p class="mut">Eye flicker (restored) vs original eye motion</p>{_spark(qc['eye_flicker'], 'var(--a)')}{_spark(qc['eye_flicker_in'], 'var(--c)')}
<p class="mut">Focus score (higher = sharper input)</p>{_spark(tracks['focus'][:n], 'var(--d)')}
<h2>Flagged ranges ({len(rs)})</h2>
<table><tr><th>from</th><th>to</th><th>frames</th><th>why</th></tr>{''.join(rows) or '<tr><td colspan=4>none</td></tr>'}</table>
<h2>Thumbnails</h2>{''.join(thumbs) or '<p class="mut">No flagged frames.</p>'}
<h2>Settings</h2><table>{s_rows}</table>
</body></html>"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
