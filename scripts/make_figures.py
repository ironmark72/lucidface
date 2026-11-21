"""Regenerate the data figures in assets/ from docs/data/*.csv.

    python scripts/make_figures.py

Palette: the validated three-slot categorical set (blue #2a78d6, orange #eb6834, aqua #1baf7a),
orange reserved for the consistency lock across all figures. Aqua sits below 3:1 contrast on the
light surface, so every aqua series carries a visible direct label.
"""
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import j1

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA, OUT = os.path.join(ROOT, "docs", "data"), os.path.join(ROOT, "assets")

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e2e1dc"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "font.family": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": "#b9b8b2",
    "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2, "savefig.dpi": 200,
})


def title(fig, t, sub):
    fig.text(0.012, 0.965, t, fontsize=15, fontweight="bold", va="top")
    fig.text(0.012, 0.905, sub, fontsize=10.5, color=INK2, va="top")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), facecolor=SURFACE)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- 1. the consistency lock
def fig_lock():
    r, s = 8.8, 5.0  # a medium defocus (radius px) and LucidFace's default lock sigma at that face size
    f = np.linspace(1e-4, 0.12, 600)
    x = 2 * np.pi * r * f
    camera = np.abs(2 * j1(x) / x)                 # disk PSF: what the lens let through
    lock = np.exp(-2 * np.pi ** 2 * s ** 2 * f ** 2)  # share of each frequency taken from the camera
    fig, ax = plt.subplots(figsize=(11, 4.6))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.78, bottom=0.14)
    title(fig, "The consistency lock: every frequency is split between the camera and the model",
          "Below the cutoff the result equals what the lens recorded (gaze, iris colour, shading). Above it, "
          "the restored detail is kept.")
    ax.fill_between(f, 0, lock, color=ORANGE, alpha=0.16, lw=0)
    ax.fill_between(f, lock, 1, color=BLUE, alpha=0.10, lw=0)
    ax.plot(f, lock, color=ORANGE)
    ax.plot(f, camera, color=INK, lw=1.6, ls=(0, (4, 3)))
    ax.text(0.006, 0.42, "taken from the camera", color=ORANGE, fontsize=11.5, fontweight="bold")
    ax.text(0.062, 0.55, "taken from the restoration model", color=BLUE, fontsize=11.5, fontweight="bold")
    ax.annotate("what the defocused lens let through\n(disk blur, radius 8.8 px)", xy=(0.092, float(np.interp(0.092, f, camera))),
                xytext=(0.086, 0.32), fontsize=10, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.annotate("lock cutoff (σ = 5 px)", xy=(0.0375, 0.5), xytext=(0.043, 0.80), fontsize=10, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.set_xlim(0, 0.12); ax.set_ylim(0, 1.02)
    ax.set_xlabel("spatial frequency (cycles per pixel)")
    ax.set_ylabel("share / response")
    save(fig, "lock_frequency.png")


# ---------------------------------------------------------------- 2. focus score separation
def fig_focus():
    rows = list(csv.DictReader(open(os.path.join(DATA, "focus_scores.csv"))))
    clips = ["potus2015", "nasa_murthy", "dvids_marceaux", "dvids_goto", "nasa_ott"]
    names = {"potus2015": "White House address", "nasa_murthy": "NASA (Murthy)", "dvids_marceaux": "DVIDS (Marceaux)",
             "dvids_goto": "DVIDS (Goto)", "nasa_ott": "NASA (Ott)"}
    fig, ax = plt.subplots(figsize=(11, 4.4))
    fig.subplots_adjust(left=0.19, right=0.97, top=0.74, bottom=0.24)
    title(fig, "The focus gate: sharp and defocused faces never overlap",
          "Focus score of the face on the five benchmark clips (sharp and blurred at 3 strengths) and on the "
          "real defocused interview. Default threshold 0.28.")
    ys = {c: i for i, c in enumerate(clips)}
    ys["real"] = len(clips) + 0.4
    ax.axvspan(0.0, 0.28, color=MUTED, alpha=0.08, lw=0)
    ax.axvline(0.28, color=INK2, lw=1.2, ls=(0, (4, 3)))
    ax.text(0.284, -0.95, "gate threshold 0.28", fontsize=10, color=INK2, va="bottom")
    ax.text(0.105, -0.95, "restored", fontsize=10, color=INK2, va="bottom")
    ax.text(0.585, -0.95, "left untouched", fontsize=10, color=INK2, va="bottom", ha="right")
    for r in rows:
        v = float(r["focus"])
        if r["condition"] == "sharp original":
            ax.scatter(v, ys[r["clip"]], s=70, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
        elif r["condition"].startswith("synthetic"):
            ax.scatter(v, ys[r["clip"]], s=70, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=3)
            if r["clip"] == "potus2015":
                ax.text(v, ys[r["clip"]] - 0.3, r["condition"][-2:], ha="center", fontsize=9, color=INK2)
        else:
            ax.scatter(v, ys["real"], s=70, color=AQUA, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.text(0.212, ys["real"], "real interview", fontsize=10, color=INK, va="center")
    ax.set_yticks([ys[c] for c in clips] + [ys["real"]], [names[c] for c in clips] + ["client footage"])
    ax.set_ylim(len(clips) + 0.9, -1.05)
    ax.set_xlim(0.1, 0.65); ax.set_xlabel("focus score (high- to mid-frequency energy of the aligned face)", labelpad=6)
    ax.grid(axis="y", visible=False)
    h = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=8) for c in (BLUE, ORANGE, AQUA)]
    fig.legend(h, ["sharp original", "synthetic defocus, levels L1 to L3", "real defocused footage"],
               loc="lower center", ncol=3, frameon=False, fontsize=10)
    save(fig, "focus_gate.png")


# ---------------------------------------------------------------- 3. benchmark results
def fig_results():
    rows = list(csv.DictReader(open(os.path.join(DATA, "results_run1.csv"))))
    cfgs = [("blurry", "blurry input", MUTED, (0, (4, 3))),
            ("cf0.7", "CodeFormer, usual setting (w = 0.7)", AQUA, "-"),
            ("cf1.0+lock+steady.25", "LucidFace default", BLUE, "-")]
    metrics = [("lpips_face", "Perceptual distance (LPIPS), lower is better"),
               ("id_cos", "Identity similarity, higher is better"),
               ("t_err_eyes", "Eye flicker vs ground truth, lower is better")]
    levels = ["1", "2", "3"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    fig.subplots_adjust(left=0.05, right=0.985, top=0.72, bottom=0.2, wspace=0.28)
    title(fig, "Benchmark: 5 public-domain interviews, 3 blur strengths",
          "Means over clips. First run: a colour bug biases restored-vs-blurry comparisons against "
          "restoration, so the restored wins are lower bounds (docs/BENCHMARK.md).")
    for ax, (m, lab) in zip(axes, metrics):
        for key, name, col, ls in cfgs:
            v = [np.mean([float(r[m]) for r in rows if r["config"] == key and r["level"] == L]) for L in levels]
            ax.plot([0, 1, 2], v, color=col, ls=ls, marker="o", markersize=7, markeredgecolor=SURFACE,
                    markeredgewidth=2)
            if m == "t_err_eyes":
                ax.text(2.08, v[-1], name.split(",")[0], fontsize=9.5, color=INK2, va="center")
        ax.set_xticks([0, 1, 2], ["mild", "medium", "heavy"])
        ax.set_xlim(-0.2, 2.2 if m != "t_err_eyes" else 3.0)
        ax.set_title(lab, fontsize=11, color=INK, loc="left", pad=10)
        ax.set_xlabel("blur strength")
    h = [plt.Line2D([], [], color=c, ls=ls, marker="o", markersize=7) for _, _, c, ls in cfgs]
    fig.legend(h, [n for _, n, _, _ in cfgs], loc="lower center", ncol=3, frameon=False, fontsize=10.5)
    save(fig, "results.png")


# ---------------------------------------------------------------- 4. QC fallback
def fig_qc():
    from lucidface import qc
    n, fps = 250, 25
    q = {"restored": np.ones(n, bool)}
    fl = {"any": np.zeros(n, bool)}
    fl["any"][96:104] = True
    fl["any"][180:182] = True
    w = qc.weights(q, fl, fps)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.7, bottom=0.17)
    title(fig, "QC fallback: flagged frames fade back to the original, never cut",
          "Example: 10 frames flagged (model departed from the footage, or eyes shimmered). "
          "The blend weight ramps over about half a second on each side.")
    for a, b in ((96, 104), (180, 182)):
        ax.axvspan(a, b, color=MUTED, alpha=0.25, lw=0)
    ax.text(100, 1.08, "flagged", ha="center", fontsize=10, color=INK2)
    ax.text(181, 1.08, "flagged", ha="center", fontsize=10, color=INK2)
    ax.plot(np.arange(n), w, color=BLUE)
    ax.text(4, 0.9, "restored", color=BLUE, fontsize=10.5, fontweight="bold")
    ax.set_ylim(-0.05, 1.18); ax.set_xlim(0, n - 1)
    ax.set_yticks([0, 1], ["original", "restored"])
    ax.set_xlabel("frame")
    save(fig, "qc_fallback.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_lock(); fig_focus(); fig_results(); fig_qc()
