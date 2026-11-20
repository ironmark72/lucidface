"""ffmpeg-based streaming video I/O. Frames are numpy uint8 BGR (H, W, 3); nothing is held in RAM
beyond one batch, so arbitrarily long videos work."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterator

import numpy as np


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: Fraction
    frames: int
    pix_fmt: str
    timecode: str | None
    has_audio: bool

    @property
    def fps_str(self) -> str:
        return f"{self.fps.numerator}/{self.fps.denominator}"


def probe(path: str) -> VideoInfo:
    out = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path]))
    v = next(s for s in out["streams"] if s["codec_type"] == "video")
    tc = (out.get("format", {}).get("tags", {}) or {}).get("timecode")
    for s in out["streams"]:
        tc = tc or (s.get("tags", {}) or {}).get("timecode")
    frames = int(v.get("nb_frames") or 0)
    if frames == 0:  # some containers do not store it: count packets (fast, no decode)
        frames = int(subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
            "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path]).strip())
    return VideoInfo(
        width=int(v["width"]), height=int(v["height"]), fps=Fraction(v["r_frame_rate"]),
        frames=frames, pix_fmt=v.get("pix_fmt", ""), timecode=tc,
        has_audio=any(s["codec_type"] == "audio" for s in out["streams"]))


class Reader:
    """Sequential frame reader. Optional crop=(x, y, w, h) is applied by ffmpeg."""

    def __init__(self, path: str, crop: tuple[int, int, int, int] | None = None,
                 start_frame: int = 0, max_frames: int | None = None, scale: float = 1.0):
        info = probe(path)
        self.info = info
        vf = []
        w, h = info.width, info.height
        if crop:
            x, y, w, h = crop
            vf.append(f"crop={w}:{h}:{x}:{y}")
        if scale != 1.0:
            w, h = int(round(w * scale / 2) * 2), int(round(h * scale / 2) * 2)
            vf.append(f"scale={w}:{h}:flags=area")
        if start_frame:
            vf.insert(0, f"select=gte(n\\,{start_frame})")
        self.w, self.h = w, h
        cmd = ["ffmpeg", "-v", "error", "-i", path]
        if vf:
            cmd += ["-vf", ",".join(vf)]
        if start_frame:
            cmd += ["-vsync", "0"]
        if max_frames:
            cmd += ["-frames:v", str(max_frames)]
        cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 3 * 8)
        self.fsize = w * h * 3

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            buf = self.proc.stdout.read(self.fsize)
            if len(buf) < self.fsize:
                break
            yield np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)
        self.close()

    def batches(self, n: int) -> Iterator[list[np.ndarray]]:
        batch = []
        for f in self:
            batch.append(f)
            if len(batch) == n:
                yield batch
                batch = []
        if batch:
            yield batch

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdout.close()
            self.proc.wait()


class Writer:
    """Raw BGR frames in, encoded video out (x264, near-lossless by default)."""

    def __init__(self, path: str, w: int, h: int, fps: str, crf: int = 10,
                 pix_fmt: str = "yuv420p", preset: str = "medium"):
        self.proc = subprocess.Popen([
            "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
            "-r", fps, "-i", "-", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", pix_fmt, "-color_primaries", "bt709", "-color_trc", "bt709",
            "-colorspace", "bt709", path], stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray):
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg encoder failed")
