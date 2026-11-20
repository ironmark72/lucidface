"""Pass 1: track one face through the whole video.

For every frame: RetinaFace 5-point landmarks (on a downscaled copy for speed), identity-by-position
tracking, gap filling, temporal smoothing that never smooths across a cut, and a scale-independent
focus score. Saved as an .npz that every later step reads.

Why smoothing matters: raw per-frame landmarks jitter by a pixel or two. The face is re-aligned from
those landmarks every frame, so jitter becomes shimmer in the restored face. Smoothing the landmarks
removes a large share of the flicker before any model runs.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from .geometry import TEMPLATE, similarity
from .videoio import Reader

DET_SIDE = 640  # detection runs with the short side at this size


def _detector(device):
    from facexlib.detection import init_detection_model
    return init_detection_model("retinaface_resnet50", half=False, device=device)


def focus_score(frame: np.ndarray, lm: np.ndarray) -> float:
    """High-frequency / mid-frequency energy of the face, measured on a 256px aligned crop so the
    number does not depend on face size. Sharp faces score high, defocused faces low."""
    m = similarity(lm)
    m = m * 0.5  # 512 template -> 256
    face = cv2.warpAffine(frame, m, (256, 256), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)
    g = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(np.float32)[48:224, 40:216]  # inner face
    b1, b2 = cv2.GaussianBlur(g, (0, 0), 1.0), cv2.GaussianBlur(g, (0, 0), 4.0)
    hi, mid = np.abs(g - b1).mean(), np.abs(b1 - b2).mean() + 1e-3
    return float(hi / mid)


def track(video: str, device: str = "cuda", stride: int = 1, prefer: str = "center",
          min_score: float = 0.8, log=print) -> dict:
    det = _detector(device)
    rd = Reader(video)
    H, W = rd.h, rd.w
    s = DET_SIDE / min(H, W) if min(H, W) > DET_SIDE else 1.0
    n = rd.info.frames
    raw = np.full((n, 5, 2), np.nan, np.float32)
    score = np.zeros(n, np.float32)
    focus = np.full(n, np.nan, np.float32)
    prev = None
    i = -1
    for i, fr in enumerate(rd):
        if i >= n:  # container under-reported the count
            raw = np.concatenate([raw, np.full((1, 5, 2), np.nan, np.float32)])
            score = np.append(score, 0); focus = np.append(focus, np.nan)
        if i % stride:
            continue
        small = cv2.resize(fr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s != 1 else fr
        with torch.no_grad():
            b = det.detect_faces(small, min_score)
        if len(b):
            b = np.asarray(b, np.float32)
            lms = b[:, 5:15].reshape(-1, 5, 2) / s
            centers = lms.mean(1)
            if prev is not None:
                k = int(np.argmin(np.linalg.norm(centers - prev, axis=1)))
            elif prefer == "largest":
                k = int(np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))
            else:
                k = int(np.argmin(np.linalg.norm(centers - [W / 2, H / 2], axis=1)))
            raw[i], score[i] = lms[k], b[k, 4]
            prev = centers[k]
            if i % 5 == 0:
                focus[i] = focus_score(fr, lms[k])
        if i % 500 == 0:
            log(f"track: {i}/{n} frames")
    n = i + 1
    raw, score, focus = raw[:n], score[:n], focus[:n]
    lm, cuts = smooth(raw)
    focus = _fill(focus)
    return dict(landmarks=lm, raw=raw, score=score, focus=focus, cuts=cuts, width=W, height=H,
                fps=float(rd.info.fps), frames=n)


def _fill(x: np.ndarray) -> np.ndarray:
    ok = ~np.isnan(x)
    if not ok.any():
        return x
    return np.interp(np.arange(len(x)), np.flatnonzero(ok), x[ok]).astype(np.float32)


def smooth(raw: np.ndarray, sigma_frames: float = 2.0, max_gap: int = 6,
           cut_iod: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    """Gap-fill (linear, up to max_gap frames) and gaussian-smooth landmarks along time.
    A jump larger than cut_iod * inter-ocular distance is treated as a cut: smoothing restarts
    there so two shots are never blended. Returns (smoothed landmarks with NaN where no face, cuts)."""
    n = len(raw)
    have = ~np.isnan(raw[:, 0, 0])
    lm = raw.copy()
    # fill short gaps
    idx = np.flatnonzero(have)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            t = (np.arange(a + 1, b) - a) / (b - a)
            lm[a + 1:b] = raw[a] * (1 - t[:, None, None]) + raw[b] * t[:, None, None]
    have = ~np.isnan(lm[:, 0, 0])
    # segments: runs of frames with a face, split at cuts
    cuts = []
    seg_start = None
    out = np.full_like(lm, np.nan)
    r = int(3 * sigma_frames)
    k = np.exp(-np.arange(-r, r + 1) ** 2 / (2 * sigma_frames ** 2)); k /= k.sum()

    def flush(a, b):
        seg = lm[a:b]
        if len(seg) <= 2:
            out[a:b] = seg
            return
        pad = np.concatenate([seg[:1].repeat(r, 0), seg, seg[-1:].repeat(r, 0)])
        flat = pad.reshape(len(pad), -1)
        sm = np.stack([np.convolve(flat[:, j], k, mode="valid") for j in range(flat.shape[1])], 1)
        out[a:b] = sm.reshape(seg.shape)

    for i in range(n):
        if not have[i]:
            if seg_start is not None:
                flush(seg_start, i); seg_start = None
            continue
        if seg_start is None:
            seg_start = i
        elif i > 0 and have[i - 1]:
            iod = np.linalg.norm(lm[i - 1, 1] - lm[i - 1, 0])
            if np.linalg.norm(lm[i].mean(0) - lm[i - 1].mean(0)) > cut_iod * iod:
                cuts.append(i)
                flush(seg_start, i); seg_start = i
    if seg_start is not None:
        flush(seg_start, n)
    # adaptive: smoothing removes jitter when the head is still, but lags behind fast turns. Where the
    # smoothed track drifts from the measurement by more than a few % of the eye distance, follow the
    # measurement instead (soft blend between 4% and 12%).
    ok = ~np.isnan(out[:, 0, 0]) & ~np.isnan(raw[:, 0, 0])
    if ok.any():
        iod = np.linalg.norm(out[ok, 1] - out[ok, 0], axis=1)
        dev = np.linalg.norm(raw[ok] - out[ok], axis=2).max(1) / np.maximum(iod, 1)
        t = np.clip((dev - 0.04) / 0.08, 0, 1)[:, None, None]
        out[ok] = out[ok] * (1 - t) + raw[ok] * t
    return out.astype(np.float32), np.array(cuts, np.int64)


def suggest_crop(tr: dict, margin: float = 1.6, multiple: int = 16) -> tuple[int, int, int, int]:
    """Smallest box that holds the aligned-face region of every tracked frame, plus margin.
    Used to upload/process only the part of the frame that can change."""
    from .geometry import face_box
    lm = tr["landmarks"]
    have = ~np.isnan(lm[:, 0, 0])
    boxes = np.array([face_box(similarity(lm[i]), tr["height"], tr["width"], pad=0)
                      for i in np.flatnonzero(have)[::5]])
    x0, y0 = boxes[:, 0].min(), boxes[:, 1].min()
    x1, y1 = boxes[:, 2].max(), boxes[:, 3].max()
    cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)
    side_w = int(np.ceil(min(tr["width"], w + (margin - 1) * (w + h) / 2) / multiple) * multiple)
    side_h = int(np.ceil(min(tr["height"], h + (margin - 1) * (w + h) / 2) / multiple) * multiple)
    side_w, side_h = min(side_w, tr["width"]), min(side_h, tr["height"])
    x = int(np.clip(cx - side_w / 2, 0, tr["width"] - side_w)) // 2 * 2
    y = int(np.clip(cy - side_h / 2, 0, tr["height"] - side_h)) // 2 * 2
    return x, y, side_w, side_h


def save(path: str, tr: dict):
    np.savez_compressed(path, **tr)


def load(path: str) -> dict:
    d = np.load(path)
    return {k: d[k] for k in d.files}
