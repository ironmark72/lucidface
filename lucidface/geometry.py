"""Face alignment on the GPU.

Faces are aligned to the standard FFHQ 5-point template at 512x512 (the space CodeFormer/GFPGAN
were trained in). Both directions (frame -> face, face -> frame) are done with torch grid_sample,
so a whole batch is warped in one call and nothing round-trips through the CPU.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

FACE = 512
# FFHQ 5-point template for 512px aligned faces: left eye, right eye, nose, mouth-left, mouth-right
TEMPLATE = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                     [201.26117, 371.41043], [313.08905, 371.15118]], np.float32)
TEMPLATE_IOD = float(np.linalg.norm(TEMPLATE[1] - TEMPLATE[0]))  # ~125.9 px


def similarity(landmarks: np.ndarray) -> np.ndarray:
    """2x3 matrix mapping frame coordinates -> 512 aligned-face coordinates."""
    m, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), TEMPLATE, method=cv2.LMEDS)
    if m is None:
        m, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), TEMPLATE)
    return m.astype(np.float64)


def invert(m: np.ndarray) -> np.ndarray:
    return cv2.invertAffineTransform(m)


def _grid(m_dst_to_src: torch.Tensor, out_h: int, out_w: int, src_h: int, src_w: int) -> torch.Tensor:
    """Sampling grid for grid_sample: for each output pixel, where to read in the source image.
    m_dst_to_src: (B, 2, 3) pixel-space affine (output pixel -> source pixel)."""
    dev, dt = m_dst_to_src.device, m_dst_to_src.dtype
    ys, xs = torch.meshgrid(torch.arange(out_h, device=dev, dtype=dt),
                            torch.arange(out_w, device=dev, dtype=dt), indexing="ij")
    pts = torch.stack([xs, ys, torch.ones_like(xs)], -1).reshape(1, -1, 3)  # (1, HW, 3)
    src = pts @ m_dst_to_src.transpose(1, 2)  # (B, HW, 2)
    gx = src[..., 0] / (src_w - 1) * 2 - 1
    gy = src[..., 1] / (src_h - 1) * 2 - 1
    return torch.stack([gx, gy], -1).reshape(-1, out_h, out_w, 2)


def warp_to_face(frames: torch.Tensor, m_frame_to_face: list[np.ndarray]) -> torch.Tensor:
    """frames (B, C, H, W) float -> aligned faces (B, C, 512, 512)."""
    B, C, H, W = frames.shape
    inv = torch.tensor(np.stack([invert(m) for m in m_frame_to_face]), device=frames.device,
                       dtype=torch.float32)
    g = _grid(inv, FACE, FACE, H, W)
    return F.grid_sample(frames, g, mode="bilinear", padding_mode="border", align_corners=True)


def warp_to_frame(faces: torch.Tensor, m_frame_to_face: list[np.ndarray], H: int, W: int,
                  box: tuple[int, int, int, int] | None = None) -> torch.Tensor:
    """aligned faces (B, C, 512, 512) -> frame space. If box=(x0, y0, x1, y1) only that region is
    produced (much cheaper than the full frame); returns (B, C, y1-y0, x1-x0)."""
    x0, y0, x1, y1 = box if box else (0, 0, W, H)
    ms = []
    for m in m_frame_to_face:
        shift = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], np.float64)  # region px -> frame px
        ms.append((np.vstack([m, [0, 0, 1]]) @ shift)[:2])
    mt = torch.tensor(np.stack(ms), device=faces.device, dtype=torch.float32)
    g = _grid(mt, y1 - y0, x1 - x0, FACE, FACE)
    return F.grid_sample(faces, g, mode="bilinear", padding_mode="zeros", align_corners=True)


def face_box(m_frame_to_face: np.ndarray, H: int, W: int, pad: int = 4) -> tuple[int, int, int, int]:
    """Bounding box in the frame of the 512 aligned square (where pasting can change pixels)."""
    inv = invert(m_frame_to_face)
    c = np.array([[0, 0, 1], [FACE, 0, 1], [0, FACE, 1], [FACE, FACE, 1]], np.float64) @ inv.T
    x0, y0 = np.floor(c.min(0)).astype(int) - pad
    x1, y1 = np.ceil(c.max(0)).astype(int) + pad
    return max(0, x0), max(0, y0), min(W, x1), min(H, y1)


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable gaussian blur on (B, C, H, W), reflect padding."""
    if sigma <= 0:
        return x
    r = int(3 * sigma + 0.5)
    k = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-(k ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    C = x.shape[1]
    x = F.pad(x, (r, r, r, r), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
    x = F.conv2d(x, k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)
    return x
