"""Command-line helpers for the event grayscale toolkit.

The CLI is intentionally auxiliary: the graphical motion-analysis player
(``event-gray-player`` / ``python -m event_grey.player``) is the main entry.
Here one can simulate a scene, run a reconstruction, print drift / static /
edge metrics and render a real reconstruction video for offline use.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .analysis import AnalysisSession
from .metrics import evaluate, format_metrics
from .player import run_server
from .plots import plot_drift, plot_error_map, plot_pixel_trace
from .simulator import SCENES, simulate
from .video import write_video, write_frame_image


def _save_dataset(bundle, outdir):
    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, "events.npy"), bundle["events"])
    np.save(
        os.path.join(outdir, "anchors.npy"),
        np.array([
            (a["t"], a["logL"], a["valid"], a["weight"])
            for a in bundle["anchors"]
        ], dtype=object),
        allow_pickle=True,
    )
    meta = {
        "scene": bundle["scene_name"],
        "shape": bundle["shape"],
        "duration": bundle["duration"],
        "contrast": bundle["contrast"],
    }
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def cmd_simulate(args):
    bundle = simulate(
        name=args.scene,
        duration=args.duration,
        contrast=args.contrast,
        anchor_every=args.anchor_every,
        anchor_coverage=args.coverage,
        corrupt_fraction=args.corrupt,
        seed=args.seed,
    )
    _save_dataset(bundle, args.outdir)
    print(f"simulated '{args.scene}': {bundle['events'].size} events, "
          f"{len(bundle['anchors'])} anchors -> {args.outdir}")


def cmd_reconstruct(args):
    bundle = simulate(
        name=args.scene, duration=args.duration, contrast=args.contrast,
        anchor_every=args.anchor_every, anchor_coverage=args.coverage,
        corrupt_fraction=args.corrupt, seed=args.seed,
    )
    session = AnalysisSession(bundle, fps=args.fps,
                              anchor_blend=args.blend,
                              reset_drift=args.reset_drift)
    result = session.result

    os.makedirs(args.outdir, exist_ok=True)
    metrics = evaluate(result, bundle["scene"])
    with open(os.path.join(args.outdir, "metrics.txt"), "w") as f:
        f.write(format_metrics(metrics) + "\n\n")
        for w in result.warnings:
            f.write(f"WARNING: {w}\n")
    print(format_metrics(metrics))
    if result.warnings:
        print("\nwarnings:")
        for w in result.warnings:
            print(" -", w)

    video_path = os.path.join(args.outdir, "reconstruction.mp4")
    write_video(video_path, result, scene=bundle["scene"],
                show_events=True, show_reference=args.with_reference,
                fps=args.fps)
    mid = result.frames[len(result.frames) // 2]
    write_frame_image(os.path.join(args.outdir, "frame_mid.png"),
                      mid[1], mid[2])
    plot_drift(result, bundle["scene"],
               os.path.join(args.outdir, "drift.png"))
    plot_error_map(result, bundle["scene"],
                   os.path.join(args.outdir, "error_map.png"))
    h, w = bundle["shape"]
    trace = session.pixel_trace(w // 2, h // 2)
    plot_pixel_trace(trace, os.path.join(args.outdir, "pixel_trace.png"))
    print(f"\nwrote {video_path}, frame_mid.png, drift.png, "
          f"error_map.png, pixel_trace.png")


def cmd_player(args):
    run_server(host=args.host, port=args.port, scene=args.scene,
               fps=args.fps, coverage=args.coverage,
               corrupt=args.corrupt, open_browser=args.browser)


def build_parser():
    p = argparse.ArgumentParser(
        prog="event-gray",
        description="Event-camera log-brightness reconstruction helpers. "
                    "The browser player is the main graphical entry point.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--scene", choices=list(SCENES), default="moving_disk")
    common.add_argument("--duration", type=float, default=2.0)
    common.add_argument("--contrast", type=float, default=0.2)
    common.add_argument("--anchor-every", type=float, default=0.25)
    common.add_argument("--coverage", type=float, default=0.15)
    common.add_argument("--corrupt", type=float, default=0.0)
    common.add_argument("--seed", type=int, default=0)

    sp = sub.add_parser("simulate", parents=[common],
                        help="simulate events + sparse anchors and save them")
    sp.add_argument("--outdir", default="out/sim")
    sp.set_defaults(func=cmd_simulate)

    rp = sub.add_parser("reconstruct", parents=[common],
                        help="run reconstruction, metrics and video export")
    rp.add_argument("--outdir", default="out/recon")
    rp.add_argument("--fps", type=float, default=20.0)
    rp.add_argument("--blend", type=float, default=0.5)
    rp.add_argument("--reset-drift", type=float, default=0.6)
    rp.add_argument("--with-reference", action="store_true",
                    help="append simulator reference panel (evaluation only)")
    rp.set_defaults(func=cmd_reconstruct)

    pp = sub.add_parser("player", parents=[common],
                        help="launch the graphical analysis player (main UI)")
    pp.add_argument("--host", default="127.0.0.1")
    pp.add_argument("--port", type=int, default=8765)
    pp.add_argument("--fps", type=float, default=20.0)
    pp.add_argument("--browser", action="store_true")
    pp.set_defaults(func=cmd_player)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
