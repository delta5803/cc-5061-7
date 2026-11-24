"""Offline matplotlib figures for reports and headless analysis.

The browser player draws its own traces on canvas; this module is used by the
CLI to persist diagnostic figures (drift curve, per-pixel integration with
anchor corrections, error map) when no GUI is available.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .metrics import evaluate
from .video import logL_to_uint8


def plot_drift(result, scene, path):
    """Mean reconstruction error vs time, with anchor times marked."""
    m = evaluate(result, scene)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(result.times(), m["drift_curve"], lw=1.6, label="mean log error")
    ax.axhline(0, color="grey", lw=0.8)
    for a in result.anchors:
        ax.axvline(a["t"], color="orange", ls="--", lw=0.8, alpha=0.7)
    failed = [r.t for r in result.reports if r.failed]
    for t in failed:
        ax.axvline(t, color="red", ls="-", lw=1.4)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mean log-brightness error")
    ax.set_title("Brightness drift (orange=anchor, red=failed anchor)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_pixel_trace(trace, path):
    """Raw event integration vs anchor-corrected trajectory for one pixel."""
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(trace["times"], trace["raw_integration"], color="#3b82f6",
            lw=1.4, label="event-only integration")
    ax.plot(trace["times"], trace["corrected"], color="#16a34a",
            lw=2.0, label="anchor-corrected")
    ev = trace["events"]
    for e in ev:
        ax.axvline(e["t"], color=("#dc2626" if e["p"] == 1 else "#2563eb"),
                   lw=0.5, alpha=0.4)
    for t, v, kind, drift in trace["anchor_points"]:
        ax.scatter([t], [v], color=("#dc2626" if kind == "reset" else "#f59e0b"),
                   zorder=5, s=28)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("log brightness")
    ax.set_title(f"Pixel ({trace['x']}, {trace['y']}) event integration "
                 f"and anchor correction ({ev.size} events)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_error_map(result, scene, path, frame_index=-1):
    """Side-by-side reconstruction, truth and signed error at one frame."""
    t = result.times()[frame_index]
    rec = result.frame_array()[frame_index]
    truth = scene.log_brightness(t)
    err = rec - truth
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, img, title in (
        (axes[0], logL_to_uint8(rec), "reconstruction"),
        (axes[1], logL_to_uint8(truth), "reference (evaluation only)"),
        (axes[2], err, "signed log error"),
    ):
        im = ax.imshow(img, cmap=("gray" if title != "signed log error" else "RdBu_r"),
                       vmin=(None if title != "signed log error" else -0.5),
                       vmax=(None if title != "signed log error" else 0.5))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        if title == "signed log error":
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"t = {t:.3f} s")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
