"""Final step: put the restoration onto the ORIGINAL camera file.

Only a box around the face is replaced; everything else is the untouched original. The box edge is
feathered, and each frame is weighted by the QC fallback weight (0 = original, 1 = restored), so
flagged frames fade back to the original. ffmpeg does all pixel work; Python only streams a small
grayscale mask per frame. Output keeps the source's bit depth/chroma (10-bit 4:2:2 stays 10-bit 4:2:2),
original audio (copied, not re-encoded) and timecode.
"""
from __future__ import annotations

import subprocess

import numpy as np

from .videoio import probe


def feather_mask(w: int, h: int, feather: int = 48) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.minimum(np.minimum(xx, w - 1 - xx), np.minimum(yy, h - 1 - yy)).astype(np.float32)
    return np.clip(d / max(feather, 1), 0, 1)


def _codec_args(pix_fmt: str, crf: int) -> list[str]:
    if "422" in pix_fmt and "10" in pix_fmt:
        return ["-c:v", "libx264", "-profile:v", "high422", "-pix_fmt", "yuv422p10le", "-crf", str(crf)]
    if "10" in pix_fmt:
        return ["-c:v", "libx264", "-profile:v", "high10", "-pix_fmt", "yuv420p10le", "-crf", str(crf)]
    if "422" in pix_fmt:
        return ["-c:v", "libx264", "-profile:v", "high422", "-pix_fmt", "yuv422p", "-crf", str(crf)]
    return ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf)]


def composite(original: str, restored: str, out: str, box: tuple[int, int, int, int],
              restored_is_crop: bool, weights: np.ndarray | None = None, feather: int = 48,
              crf: int = 12, preset: str = "fast", log=print):
    """box=(x, y, w, h) in original-frame pixels. restored_is_crop: the restored video is exactly that
    box (crop workflow); otherwise it is full-frame and gets cropped to the box here."""
    info = probe(original)
    x, y, w, h = box
    n = info.frames
    wts = np.ones(n, np.float32) if weights is None else np.asarray(weights, np.float32)
    base = feather_mask(w, h, feather)
    src = "[1:v]" if restored_is_crop else f"[1:v]crop={w}:{h}:{x}:{y},"
    fmt = "yuv444p10le" if "10" in info.pix_fmt else "yuv444p"
    fc = (f"{src}format={fmt}[c];[2:v]format=gray[m];[c][m]alphamerge[ca];"
          f"[0:v][ca]overlay={x}:{y}:shortest=1:format=auto[v]")
    cmd = ["ffmpeg", "-v", "error", "-stats", "-y", "-i", original, "-i", restored,
           "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", info.fps_str, "-i", "-",
           "-filter_complex", fc, "-map", "[v]"]
    if info.has_audio:
        cmd += ["-map", "0:a", "-c:a", "copy"]
    cmd += _codec_args(info.pix_fmt, crf) + ["-preset", preset, "-x264-params", "keyint=50",
                                             "-color_primaries", "bt709", "-color_trc", "bt709",
                                             "-colorspace", "bt709", "-map_metadata", "0"]
    if info.timecode:
        cmd += ["-timecode", info.timecode]
    cmd.append(out)
    log("composite: " + " ".join(cmd[:12]) + " ...")
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        cache = {}
        for i in range(n):
            wv = float(wts[i]) if i < len(wts) else 0.0
            key = round(wv, 3)
            m = cache.get(key)
            if m is None:
                m = (base * key * 255 + 0.5).astype(np.uint8).tobytes()
                if len(cache) < 64:
                    cache[key] = m
            p.stdin.write(m)
        p.stdin.close()
    except BrokenPipeError:
        pass
    if p.wait() != 0:
        raise RuntimeError("ffmpeg composite failed")
