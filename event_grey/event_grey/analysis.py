"""Analysis backend for the motion-analysis player.

A single :class:`AnalysisSession` owns one simulated (or loaded) dataset and
serves every GUI request: time-window event distributions, reconstructed
frames at arbitrary times, per-pixel integration traces and the list of
anomalies (anchor failures, drift, unsupported regions).  All numbers shown
in the interface come from the actual reconstruction computation here.
"""

from __future__ import annotations

import numpy as np

from .model import make_prior
from .reconstruct import Reconstructor
from .simulator import simulate
from .video import event_window_image, logL_to_uint8, unsupported_overlay


class AnalysisSession:
    def __init__(self, bundle, contrast=None, anchor_blend=0.5,
                 reset_drift=0.6, anchor_fail_drift=1.5, fps=20.0):
        self.bundle = bundle
        self.scene = bundle["scene"]
        self.events = bundle["events"]
        self.anchors = bundle["anchors"]
        self.shape = bundle["shape"]
        self.contrast = contrast or bundle["contrast"]
        self.duration = bundle["duration"]
        self.fps = fps

        self.reconstructor = Reconstructor(
            self.shape,
            contrast=self.contrast,
            anchor_blend=anchor_blend,
            reset_drift=reset_drift,
            anchor_fail_drift=anchor_fail_drift,
            prior=make_prior(self.shape),
        )
        for a in self.anchors:
            self.reconstructor.add_anchor(a)

        self.output_times = np.arange(0.0, self.duration + 1e-9, 1.0 / fps)
        self.result = self.reconstructor.run(self.events, self.output_times)
        self._rebuild_diagnostics()

    # ------------------------------------------------------------------ #
    @classmethod
    def from_simulation(cls, scene_name="moving_disk", fps=20.0, **kwargs):
        bundle = simulate(name=scene_name, **kwargs)
        return cls(bundle, fps=fps)

    # ------------------------------------------------------------------ #
    def _rebuild_diagnostics(self):
        res = self.result
        self.anomalies = []
        for w in res.warnings:
            self.anomalies.append({"level": "warn", "message": w})
        for rep in res.reports:
            if rep.failed:
                self.anomalies.append({
                    "level": "error",
                    "message": (
                        f"Anchor at t={rep.t:.3f}s FAILED: {rep.reason}. "
                        "Frame rejected; integration continues without it."
                    ),
                })
            elif rep.global_drift != 0.0 and abs(rep.global_drift) > 0.4:
                # The first anchor replaces the neutral prior rather than
                # correcting a measured drift; only flag later anchors.
                earlier = [r for r in res.reports
                           if r.t < rep.t and not r.failed]
                if earlier:
                    self.anomalies.append({
                        "level": "warn",
                        "message": (
                            f"Anchor t={rep.t:.3f}s corrected median drift "
                            f"{rep.global_drift:+.3f} log units "
                            f"({rep.reset_pixels} pixels hard-reset)"
                        ),
                    })
        unsupported = int((~res.supported_masks()[-1]).sum())
        if unsupported:
            self.anomalies.append({
                "level": "warn",
                "message": (
                    f"{unsupported} pixels have no anchor/prior support: "
                    "neutral-grey prior shown, not measured brightness"
                ),
            })
        if self.events.size == 0:
            self.anomalies.append({
                "level": "info",
                "message": "Event stream is empty; only anchor data is shown",
            })

    # ------------------------------------------------------------------ #
    def frame_index(self, t):
        idx = int(np.searchsorted(self.output_times, t, side="left"))
        return int(np.clip(idx, 0, len(self.output_times) - 1))

    def event_window(self, t0, t1):
        """Actual per-pixel event statistics over [t0, t1)."""
        m = (self.events["t"] >= t0) & (self.events["t"] < t1)
        ev = self.events[m]
        h, w = self.shape
        on_count = np.zeros((h, w), dtype=np.int64)
        off_count = np.zeros((h, w), dtype=np.int64)
        if ev.size:
            on = ev[ev["p"] == 1]
            off = ev[ev["p"] == 0]
            np.add.at(on_count, (on["y"], on["x"]), 1)
            np.add.at(off_count, (off["y"], off["x"]), 1)
        return {
            "t0": float(t0),
            "t1": float(t1),
            "n_events": int(ev.size),
            "n_on": int((ev["p"] == 1).sum()) if ev.size else 0,
            "n_off": int((ev["p"] == 0).sum()) if ev.size else 0,
            "on_count": on_count,
            "off_count": off_count,
            "net": on_count - off_count,
            "image": event_window_image(self.events, t0, t1, self.shape),
        }

    def reconstructed_frame(self, t):
        idx = self.frame_index(t)
        ft, logL, supported = self.result.frames[idx]
        grey = logL_to_uint8(logL)
        return {
            "t": ft,
            "index": idx,
            "grey": grey,
            "rgb": unsupported_overlay(grey, supported),
            "supported": supported,
            "logL": logL,
        }

    def reference_frame(self, t):
        """True frame for the optional evaluation view; never used in recon."""
        return logL_to_uint8(self.scene.log_brightness(t))

    def pixel_trace(self, x, y):
        return self.reconstructor.pixel_trace(
            int(x), int(y), self.events, self.output_times
        )

    def pixel_events(self, x, y):
        m = (self.events["x"] == x) & (self.events["y"] == y)
        return self.events[m]

    def anchor_summary(self):
        return [
            {
                "t": r.t,
                "failed": r.failed,
                "global_drift": r.global_drift,
                "reset_pixels": r.reset_pixels,
                "blended_pixels": r.blended_pixels,
                "supported_pixels": r.supported_pixels,
                "reason": r.reason,
            }
            for r in self.result.reports
        ]
