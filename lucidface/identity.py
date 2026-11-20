"""Reference-photo identity lock.

The user supplies one or more sharp photos of the same person (or a frame from an in-focus second
camera). Each frame is restored several ways (e.g. CodeFormer at fidelity 1.0 / 0.8 / 0.6, all under
the consistency lock), every candidate is embedded with a face-recognition network, and the one closest
to the reference identity is kept. Scores are smoothed over time with hysteresis so the choice does not
flip frame to frame. The identity similarity also goes into the QC report, so frames where the face
drifts away from the real person are flagged even when no candidate is good.

Identity network: facenet-pytorch InceptionResnetV1 (code MIT; weights trained on VGGFace2: check the
dataset terms before commercial use). Install with `pip install --no-deps facenet-pytorch`.
Published reference-based restorers (InstantRestore, DMDNet, ASFFNet, RestorerID, ReF-LDM) all have
non-commercial or missing licences as of 2026-11, so they are not the default; see docs/ROADMAP.md.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from . import geometry as G

# region of the 512 aligned face fed to the identity net (tight face, like an MTCNN crop)
ID_BOX = (88, 104, 424, 440)  # x0, y0, x1, y1


class Embedder:
    def __init__(self, device):
        from facenet_pytorch import InceptionResnetV1
        self.net = InceptionResnetV1(pretrained="vggface2").to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, faces01: torch.Tensor) -> torch.Tensor:
        """aligned faces (B,3,512,512) RGB [0,1] -> unit embeddings (B,512)."""
        x0, y0, x1, y1 = ID_BOX
        x = F.interpolate(faces01[..., y0:y1, x0:x1], size=(160, 160), mode="bilinear",
                          align_corners=False)
        x = (x * 255 - 127.5) / 128.0
        return F.normalize(self.net(x), dim=1)


def reference_embedding(paths: list[str], embedder: Embedder, device) -> torch.Tensor:
    from facexlib.detection import init_detection_model
    det = init_detection_model("retinaface_resnet50", half=False, device=device)
    embs = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(p)
        with torch.no_grad():
            b = det.detect_faces(img, 0.9)
        if not len(b):
            raise ValueError(f"no face found in reference {p}")
        b = max(b, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
        lm = np.asarray(b[5:15], np.float32).reshape(5, 2)
        m = G.similarity(lm)
        face = cv2.warpAffine(img, m, (G.FACE, G.FACE), flags=cv2.INTER_AREA,
                              borderMode=cv2.BORDER_REFLECT)
        t = torch.from_numpy(face[..., ::-1].copy()).permute(2, 0, 1)[None].float().to(device) / 255
        embs.append(embedder(t))
    return F.normalize(torch.cat(embs).mean(0, keepdim=True), dim=1)


class IdentityLock:
    def __init__(self, refs: list[str], variants: list, device="cuda", smooth: float = 0.8,
                 hysteresis: float = 0.01):
        """variants: list of (label, restorer) to choose between, e.g. CodeFormer at several w."""
        self.embed = Embedder(device)
        self.ref = reference_embedding(refs, self.embed, device)
        self.variants = variants
        self.smooth, self.hyst = smooth, hysteresis
        self.ema = None
        self.cur = 0

    def reset(self):
        self.ema = None

    def similarity(self, faces01: torch.Tensor) -> torch.Tensor:
        return (self.embed(faces01) @ self.ref.T).squeeze(1)

    def pick(self, sims: np.ndarray) -> int:
        """sims: similarity of each variant for one frame. Returns the chosen variant index."""
        self.ema = sims if self.ema is None else self.smooth * self.ema + (1 - self.smooth) * sims
        best = int(np.argmax(self.ema))
        if self.ema[best] > self.ema[self.cur] + self.hyst:
            self.cur = best
        return self.cur
