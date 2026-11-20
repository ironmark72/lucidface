import numpy as np
import torch

from lucidface import geometry as G
from lucidface import qc, track


def _lm(cx=400.0, cy=300.0, s=1.0):
    t = (G.TEMPLATE - G.TEMPLATE.mean(0)) * 0.4 * s
    return (t + [cx, cy]).astype(np.float32)


def test_align_roundtrip():
    H, W = 600, 800
    torch.manual_seed(0)
    img = G.gaussian_blur(torch.rand(1, 3, H, W), 3.0)
    m = G.similarity(_lm())
    face = G.warp_to_face(img, [m])
    assert face.shape == (1, 3, 512, 512)
    x0, y0, x1, y1 = G.face_box(m, H, W)
    back = G.warp_to_frame(face, [m], H, W, box=(x0, y0, x1, y1))
    # compare the inner part of the box (edges of the 512 square are partially outside)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    r = (y1 - y0) // 4
    a = img[..., cy - r:cy + r, cx - r:cx + r]
    b = back[..., cy - r - y0:cy + r - y0, cx - r - x0:cx + r - x0]
    assert (a - b).abs().mean() < 0.01


def test_consistency_lock_pins_low_frequencies():
    torch.manual_seed(0)
    a = G.gaussian_blur(torch.rand(1, 3, 128, 128), 4.0)          # "observed" blurry face
    r = torch.rand(1, 3, 128, 128)                                # arbitrary model output
    s = 6.0
    out = r + G.gaussian_blur(a - r, s)
    # after the lock, the low-pass of the output equals the low-pass of the observation...
    lo = (G.gaussian_blur(out, 2 * s) - G.gaussian_blur(a, 2 * s)).abs().mean()
    assert lo < 0.02
    # ...while the high frequencies of the model output survive
    hi_r = r - G.gaussian_blur(r, s)
    hi_o = out - G.gaussian_blur(out, s)
    assert torch.corrcoef(torch.stack([hi_r.flatten(), hi_o.flatten()]))[0, 1] > 0.9


def test_smoothing_fills_gaps_and_respects_cuts():
    n = 60
    raw = np.stack([_lm(400 + (i >= 30) * 300, 300) for i in range(n)])  # cut at frame 30
    raw = raw + np.random.default_rng(0).normal(0, 1.0, raw.shape).astype(np.float32)
    raw[10:13] = np.nan                                                  # short gap
    lm, cuts = track.smooth(raw)
    assert 30 in cuts.tolist()
    assert not np.isnan(lm[11]).any()
    # no blending across the cut: frame 29 stays near x=400, frame 30 near x=700
    assert abs(lm[29].mean(0)[0] - 400) < 5 and abs(lm[30].mean(0)[0] - 700) < 5
    # jitter reduced
    assert np.std(np.diff(lm[35:55, 0, 0])) < np.std(np.diff(raw[35:55, 0, 0]))


def test_focus_gate_hysteresis_and_min_length():
    f = np.r_[np.full(100, 0.5), np.full(100, 2.0), np.full(5, 0.5), np.full(100, 2.0)]
    g = qc.focus_gate(f, fps=25, thr=1.0)
    assert g[:90].all() and not g[110:].any()  # the 5-frame dip is too short to switch on


def test_fallback_weights_ramp():
    n = 100
    q = {"restored": np.ones(n, bool)}
    fl = {"any": np.zeros(n, bool)}
    fl["any"][50] = True
    w = qc.weights(q, fl, fps=25)
    assert w[50] == 0 and w[0] == 1 and w[99] == 1
    assert 0 < w[42] < 1 or 0 < w[43] < 1  # gradual, not a hard switch
