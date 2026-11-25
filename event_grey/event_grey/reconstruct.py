"""Event integration and anchor-corrected reconstruction.

The reconstructor sweeps the event stream in physical time order.  Every
event adds a signed log-brightness increment (``+C`` for ON, ``-C`` for
OFF) to its pixel, so intermediate intensity peaks survive even when the
net event count over a window is zero -- the events are integrated in time
order, never reduced to a positive/negative count subtraction.

Absolute brightness is only introduced through explicit anchors or the
caller-provided prior.  Anchors are applied with a soft blend while the
drift accumulated since the previous anchor stays within a plausible band;
once the drift exceeds that band the pixel is hard-reset to the anchor,
because event integration has demonstrably lost the absolute level.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import (
    NEUTRAL_GREY,
    make_prior,
    sort_events,
)


@dataclass
class AnchorReport:
    """What happened when one anchor was applied."""

    t: float
    supported_pixels: int
    reset_pixels: int
    blended_pixels: int
    global_drift: float
    failed: bool = False
    reason: str = ""


@dataclass
class ReconstructionResult:
    """Output of a reconstruction sweep."""

    frames: list  # list of (t, logL, supported)
    events: np.ndarray
    anchors: list
    reports: list
    warnings: list = field(default_factory=list)
    shape: tuple = (0, 0)
    contrast: float = 0.0

    def frame_array(self):
        """Stack all reconstructed log-brightness frames (T, H, W)."""
        return np.stack([f[1] for f in self.frames], axis=0)

    def times(self):
        return np.array([f[0] for f in self.frames], dtype=np.float64)

    def supported_masks(self):
        return np.stack([f[2] for f in self.frames], axis=0)

    def intensity_frames(self):
        """Convert reconstructed log brightness back to intensity in [0, 1]."""
        return np.exp(self.frame_array()) - 1e-3


class Reconstructor:
    """Streaming event integrator with anchor correction.

    Parameters
    ----------
    shape:
        Sensor shape (height, width).
    contrast:
        Log-brightness contrast threshold C emitted per event.
    anchor_blend:
        Weight of the anchor in a soft blend (1.0 = trust anchor fully).
    reset_drift:
        Drift (in log units) above which a pixel is hard-reset instead of
        blended, i.e. the accumulated integration no longer agrees with the
        anchor closely enough to be salvageable.
    anchor_fail_drift:
        Median drift above which an entire anchor is declared failed
        (corrupt / mis-registered).  Failed anchors are ignored and an
        explicit warning is emitted.
    """

    def __init__(
        self,
        shape,
        contrast=0.2,
        anchor_blend=0.5,
        reset_drift=0.6,
        anchor_fail_drift=1.5,
        prior=None,
    ):
        self.shape = (int(shape[0]), int(shape[1]))
        if contrast <= 0:
            raise ValueError("contrast threshold must be positive")
        self.contrast = float(contrast)
        self.anchor_blend = float(anchor_blend)
        self.reset_drift = float(reset_drift)
        self.anchor_fail_drift = float(anchor_fail_drift)

        prior = prior or make_prior(self.shape)
        self.logL = prior["logL"].copy()
        self.supported = np.array(prior["supported"], dtype=bool, copy=True)
        # The starting state before any event or anchor.  ``pixel_trace`` must
        # begin from this exact prior, not a hard-coded neutral grey, or the
        # per-pixel plot would disagree with the reconstructed frames.
        self._prior_logL = self.logL.copy()
        self._prior_supported = self.supported.copy()
        # Number of events integrated per pixel (for diagnostics/UI).
        self.event_count = np.zeros(self.shape, dtype=np.int64)
        self.net_polarity = np.zeros(self.shape, dtype=np.int64)
        self.last_event_t = np.full(self.shape, -np.inf, dtype=np.float64)

        self.anchors = []
        self.reports = []
        self.warnings = []
        self._cursor = 0  # index of next unprocessed event
        self._last_anchor_t = None

    # ------------------------------------------------------------------ #
    def add_anchor(self, anchor):
        """Register an absolute-brightness anchor (frame or future frame)."""
        if anchor["logL"].shape != self.shape:
            raise ValueError("anchor shape does not match sensor shape")
        self.anchors.append(dict(anchor))
        self.anchors.sort(key=lambda a: a["t"])

    # ------------------------------------------------------------------ #
    def _anchor_effect(self, anchor, logL, supported):
        """Apply one anchor to ``(logL, supported)`` in place.

        This is the single source of truth for anchor semantics: the
        streaming sweep (``_apply_anchor``) and the per-pixel trace replay
        (``pixel_trace``) both call it, so the player can never disagree with
        the frames it renders.  Returns ``(report, kinds)`` where ``kinds``
        is a per-pixel array of ``"blend"/"reset"/"first"/""`` for a
        successful anchor, or ``None`` when the whole anchor is rejected.
        """
        valid = anchor["valid"]
        target = anchor["logL"]
        n_supported = int(valid.sum())
        if n_supported == 0:
            return AnchorReport(
                t=anchor["t"],
                supported_pixels=0,
                reset_pixels=0,
                blended_pixels=0,
                global_drift=0.0,
                failed=True,
                reason="anchor has no supported pixels",
            ), None

        drift = target - logL
        observed = drift[valid]
        global_drift = float(np.median(observed))

        # A gross median mismatch means the anchor itself is untrustworthy
        # (corruption / dropped sync).  Rejecting it keeps a bad frame from
        # poisoning every pixel; the next good anchor will reveal the drift.
        if abs(global_drift) > self.anchor_fail_drift:
            return AnchorReport(
                t=anchor["t"],
                supported_pixels=n_supported,
                reset_pixels=0,
                blended_pixels=0,
                global_drift=global_drift,
                failed=True,
                reason=f"median drift {global_drift:.3f} exceeds "
                f"failure band {self.anchor_fail_drift:.3f}",
            ), None

        reset_mask = valid & (np.abs(drift) > self.reset_drift)
        # A pixel meeting its first anchor has no measured level yet: the
        # neutral prior is not information, so take the anchor fully.
        first_mask = valid & ~supported
        blend_mask = valid & ~reset_mask & ~first_mask

        w = self.anchor_blend * float(anchor.get("weight", 1.0))
        w = min(max(w, 0.0), 1.0)
        if np.any(blend_mask):
            logL[blend_mask] = (1.0 - w) * logL[blend_mask] + w * target[blend_mask]
        if np.any(reset_mask | first_mask):
            logL[reset_mask | first_mask] = target[reset_mask | first_mask]
        supported[valid] = True

        kinds = np.full(valid.shape, "", dtype=object)
        kinds[blend_mask] = "blend"
        kinds[reset_mask] = "reset"
        kinds[first_mask] = "first"

        report = AnchorReport(
            t=anchor["t"],
            supported_pixels=n_supported,
            reset_pixels=int(reset_mask.sum()),
            blended_pixels=int(blend_mask.sum()) + int(first_mask.sum()),
            global_drift=global_drift,
        )
        return report, kinds

    def _apply_anchor(self, anchor):
        report, _ = self._anchor_effect(anchor, self.logL, self.supported)
        self.reports.append(report)
        if report.failed:
            if report.supported_pixels == 0:
                self.warnings.append(
                    f"t={anchor['t']:.4f}s: anchor rejected (empty support mask)"
                )
            else:
                self.warnings.append(
                    f"t={anchor['t']:.4f}s: anchor FAILED "
                    f"(median drift {report.global_drift:+.3f} log units) - "
                    f"ignored, relying on event integration until next anchor"
                )
            return report
        self._last_anchor_t = anchor["t"]
        return report

    # ------------------------------------------------------------------ #
    def _integrate(self, events):
        """Apply a contiguous block of events in array order."""
        if events.size == 0:
            return
        inc = np.where(events["p"] == 1, self.contrast, -self.contrast)
        np.add.at(self.logL, (events["y"], events["x"]), inc)
        np.add.at(self.event_count, (events["y"], events["x"]), 1)
        np.add.at(
            self.net_polarity,
            (events["y"], events["x"]),
            events["p"].astype(np.int64) * 2 - 1,
        )
        np.maximum.at(
            self.last_event_t, (events["y"], events["x"]), events["t"]
        )

    def _integrate_logL(self, logL, events):
        """Pure signed-log-brightness integration into a caller-owned array.

        Used by ``pixel_trace`` so its replay shares the exact integration
        increment with the streaming sweep without touching live diagnostics.
        """
        if events.size == 0:
            return
        inc = np.where(events["p"] == 1, self.contrast, -self.contrast)
        np.add.at(logL, (events["y"], events["x"]), inc)

    # ------------------------------------------------------------------ #
    def run(self, events, output_times):
        """Sweep ``events`` and emit reconstructed frames at output times.

        The output time grid has no effect on the accumulated physics: the
        same event blocks are integrated regardless of when frames are
        sampled; changing ``output_times`` only changes where the continuous
        integration is snapshotted.
        """
        events = sort_events(events)
        output_times = np.sort(np.asarray(output_times, dtype=np.float64))

        t_start = output_times[0] if output_times.size else -np.inf
        if events.size and events[0]["t"] < t_start:
            self.warnings.append(
                f"{int((events['t'] < t_start).sum())} events precede the "
                "first output time and are integrated before the first frame"
            )

        anchor_queue = sorted(self.anchors, key=lambda a: a["t"])
        anchor_idx = 0
        frames = []

        for t in output_times:
            # Anchors at or before this output time are applied at their
            # own timestamp, interleaved with the event blocks.
            while anchor_idx < len(anchor_queue) and anchor_queue[anchor_idx]["t"] <= t:
                anchor = anchor_queue[anchor_idx]
                end = np.searchsorted(events["t"], anchor["t"], side="right")
                if end > self._cursor:
                    self._integrate(events[self._cursor:end])
                    self._cursor = end
                self._apply_anchor(anchor)
                anchor_idx += 1

            end = np.searchsorted(events["t"], t, side="right")
            if end > self._cursor:
                self._integrate(events[self._cursor:end])
                self._cursor = end

            frames.append((float(t), self.logL.copy(), self.supported.copy()))

        # Drain events/anchors after the last requested output time so that
        # diagnostics stay complete even though no frame is emitted there.
        while anchor_idx < len(anchor_queue):
            anchor = anchor_queue[anchor_idx]
            end = np.searchsorted(events["t"], anchor["t"], side="right")
            if end > self._cursor:
                self._integrate(events[self._cursor:end])
                self._cursor = end
            self._apply_anchor(anchor)
            anchor_idx += 1
        if events.size > self._cursor:
            self._integrate(events[self._cursor:])
            self._cursor = events.size

        if not self.supported.all():
            n_unsup = int((~self.supported).sum())
            self.warnings.append(
                f"{n_unsup} pixels never covered by an anchor or supported "
                f"prior - shown with explicit neutral-grey prior "
                f"(L={NEUTRAL_GREY:.3f}), not measured brightness"
            )

        return ReconstructionResult(
            frames=frames,
            events=events,
            anchors=list(self.anchors),
            reports=list(self.reports),
            warnings=list(self.warnings),
            shape=self.shape,
            contrast=self.contrast,
        )

    # ------------------------------------------------------------------ #
    def pixel_trace(self, x, y, events, output_times):
        """Per-pixel event integration trace with anchor corrections.

        Returns a dict with the event list at the pixel, the raw event-only
        log-brightness trajectory, the anchor-corrected trajectory and the
        anchor correction points.  This is exactly what the GUI plots for
        the selected pixel.

        The corrected trajectory is produced by replaying the *same* sweep
        the reconstructed frames use -- anchors applied at their own
        timestamps, events interleaved in between, failed anchors rejected
        and the caller's actual prior as the starting state.  The plotted
        value at each output time therefore equals ``frame[y, x]`` of the
        reconstruction at that time; it is not a second, independently
        ordered integration.
        """
        events = sort_events(events)
        m = (events["x"] == x) & (events["y"] == y)
        pe = events[m]

        raw = np.zeros_like(output_times, dtype=np.float64)
        level = 0.0
        j = 0
        for i, t in enumerate(output_times):
            while j < pe.size and pe[j]["t"] <= t:
                level += self.contrast if pe[j]["p"] == 1 else -self.contrast
                j += 1
            raw[i] = level

        # Replay the reconstruction sweep on a private copy of the frame
        # state so the trace matches the emitted frames exactly.
        logL = self._prior_logL.copy()
        supported = self._prior_supported.copy()
        corrected = np.full_like(output_times, np.nan, dtype=np.float64)
        anchor_points = []

        anchor_queue = sorted(self.anchors, key=lambda a: a["t"])
        anchor_idx = 0
        cursor = 0  # next unprocessed event index

        for i, t in enumerate(output_times):
            # Anchors at or before this output time are applied at their own
            # timestamp, after integrating the events that precede them.
            while (anchor_idx < len(anchor_queue)
                   and anchor_queue[anchor_idx]["t"] <= t):
                anchor = anchor_queue[anchor_idx]
                end = np.searchsorted(events["t"], anchor["t"], side="right")
                if end > cursor:
                    self._integrate_logL(logL, events[cursor:end])
                    cursor = end
                drift = float(anchor["logL"][y, x]) - float(logL[y, x])
                report, kinds = self._anchor_effect(anchor, logL, supported)
                if kinds is not None and anchor["valid"][y, x]:
                    anchor_points.append((
                        anchor["t"], float(logL[y, x]),
                        kinds[y, x], drift,
                    ))
                anchor_idx += 1

            end = np.searchsorted(events["t"], t, side="right")
            if end > cursor:
                self._integrate_logL(logL, events[cursor:end])
                cursor = end
            corrected[i] = logL[y, x]

        return {
            "x": x,
            "y": y,
            "events": pe,
            "times": np.asarray(output_times, dtype=np.float64),
            "raw_integration": raw + float(self._prior_logL[y, x]),
            "corrected": corrected,
            "anchor_points": anchor_points,
        }
