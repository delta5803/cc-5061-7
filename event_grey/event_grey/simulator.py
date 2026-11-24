"""Continuous-scene simulator producing events, reference frames and anchors.

Every scene is a deterministic function of time ``I(x, y, t)`` returning an
intensity in [0, 1].  Events are generated from the *threshold-crossing
model*: per pixel, log brightness is integrated between simulation steps and
an ON/OFF event is emitted each time the accumulated change reaches the
contrast threshold, exactly mirroring the model declared in
:mod:`event_grey.model`.

Reference frames are the simulator's true intensity/log-brightness field at
chosen times.  They are returned separately and are used only for (a)
building genuinely sparse anchors and (b) offline error evaluation -- the
reconstructor never receives them.
"""

from __future__ import annotations

import numpy as np

from .model import LOG_EPS, events_from_arrays, make_anchor


class Scene:
    """Abstract continuous scene."""

    def intensity(self, t):
        raise NotImplementedError

    def log_brightness(self, t):
        return np.log(self.intensity(t) + LOG_EPS)

    def log_brightness_at_times(self, times):
        """Stack log-brightness frames for an array of times."""
        return np.stack([self.log_brightness(float(t)) for t in times], axis=0)


class MovingDiskScene(Scene):
    """A bright disk moving over a static gradient background.

    The disk sweeps the field, creating fast moving edges while the
    background stays perfectly still -- a direct test of both edge
    reconstruction and the no-event / static-region requirement.
    """

    def __init__(self, shape=(64, 80), radius=10, speed=22.0,
                 disk_i=0.9, bg_min=0.15, bg_max=0.45):
        self.shape = shape
        self.radius = radius
        self.speed = speed
        self.disk_i = disk_i
        self.bg_min, self.bg_max = bg_min, bg_max
        h, w = shape
        yy, xx = np.mgrid[0:h, 0:w]
        t = (xx - (w - 1) / 2.0) / max(w - 1, 1)
        self._bg = bg_min + (bg_max - bg_min) * t
        self._yy, self._xx = yy, xx

    def intensity(self, t):
        h, w = self.shape
        cx = self.radius + 2 + self.speed * t
        cy = h * 0.5 + 6.0 * np.sin(2 * np.pi * t / 2.0)
        inside = (self._xx - cx) ** 2 + (self._yy - cy) ** 2 <= self.radius ** 2
        return np.where(inside, self.disk_i, self._bg)


class DriftingGratingScene(Scene):
    """A sinusoidal grating drifting at constant velocity.

    Pixels see smooth log-brightness oscillation (peaks *and* troughs), so
    windows with equal ON/OFF counts but nonzero intermediate intensity can
    be tested directly.
    """

    def __init__(self, shape=(64, 80), period=18.0, speed=14.0,
                 mean=0.45, amp=0.35):
        self.shape = shape
        self.period, self.speed, self.mean, self.amp = period, speed, mean, amp
        h, w = shape
        yy, xx = np.mgrid[0:h, 0:w]
        self._xx = xx

    def intensity(self, t):
        phase = 2 * np.pi * (self._xx - self.speed * t) / self.period
        return np.clip(self.mean + self.amp * np.cos(phase), 0.02, 0.98)

class StaticPanelScene(Scene):
    """A time-invariant panel with sharp edges.

    Emits no events at all after the initial settle; used to prove that
    regions without events stay exactly at their anchored level and never
    drift, and that an unsupported region keeps the explicit prior.
    """

    def __init__(self, shape=(64, 80)):
        self.shape = shape
        h, w = shape
        img = np.full(shape, 0.2)
        img[:, w // 3: 2 * w // 3] = 0.5
        img[: h // 2, :] = img[: h // 2, :] + 0.1
        self._img = img

    def intensity(self, t):
        return self._img.copy()


class FastEdgeScene(Scene):
    """A sharp vertical edge that jumps rapidly across the sensor.

    Combines a slow brightening ramp (global drift) with a high-velocity
    edge to stress fast-edge reconstruction and anchor drift detection.
    """

    def __init__(self, shape=(64, 80), speed=45.0, dark=0.08, bright=0.85):
        self.shape = shape
        self.speed, self.dark, self.bright = speed, dark, bright
        h, w = shape
        yy, xx = np.mgrid[0:h, 0:w]
        self._xx = xx

    def intensity(self, t):
        edge_x = 4 + self.speed * t
        ramp = 0.15 * t
        base = np.where(self._xx >= edge_x, self.bright, self.dark)
        return np.clip(base + ramp, 0.0, 1.0)


SCENES = {
    "moving_disk": MovingDiskScene,
    "grating": DriftingGratingScene,
    "static_panel": StaticPanelScene,
    "fast_edge": FastEdgeScene,
}


def generate_events(scene, duration, contrast=0.2, dt=1e-3, t0=0.0, seed=0,
                    threshold_jitter=0.0):
    """Threshold-crossing event generation for a continuous scene.

    The simulator keeps, per pixel, the log brightness at which the last
    event was emitted.  Between time steps the accumulated log change is
    consumed in whole threshold steps.  Multiple threshold crossings within
    one simulation step produce multiple events (each at the current step
    time), matching real sensor behaviour under fast motion.

    ``threshold_jitter`` (fractional standard deviation of the per-pixel
    threshold) defaults to zero: the declared model uses a single fixed
    contrast C, and the reconstructor integrates exactly that C.  Enable
    jitter only to study threshold-mismatch drift explicitly.
    """
    rng = np.random.RandomState(seed)
    h, w = scene.shape
    # Per-pixel level at which the last event fired.
    L_ref = scene.log_brightness(t0)
    # Per-pixel threshold; zero jitter reproduces the declared fixed-C model.
    c_map = contrast * np.exp(threshold_jitter * rng.randn(h, w))

    ts, xs, ys, ps = [], [], [], []
    n_steps = int(round(duration / dt))
    for k in range(1, n_steps + 1):
        t = t0 + k * dt
        L = scene.log_brightness(t)
        dL = L - L_ref
        # Number of whole threshold steps crossed in each direction.  A
        # large fast motion can cross several thresholds within one step;
        # each crossing emits one event and advances the reference level.
        n_on = np.floor(np.maximum(dL, 0.0) / c_map).astype(np.int64)
        n_off = np.floor(np.maximum(-dL, 0.0) / c_map).astype(np.int64)

        # Emit each threshold crossing as its own event.  Several crossings
        # of one pixel inside a single simulation step are spread across
        # ordered sub-step timestamps: a real sensor cannot emit two
        # same-polarity events at the identical instant, and the canonical
        # event stream treats (t, x, y, p) duplicates as repeats.
        def emit(count_map, polarity):
            yy, xx = np.nonzero(count_map > 0)
            for y, x in zip(yy, xx):
                n = int(count_map[y, x])
                for j in range(n):
                    ts.append(t + (j - n + 1) * dt * 1e-3)
                    xs.append(int(x))
                    ys.append(int(y))
                    ps.append(polarity)

        emit(n_on, 1)
        emit(n_off, 0)

        # Advance each reference by exactly the emitted threshold steps so
        # the residual sub-threshold change carries into the next step.
        L_ref += np.where(n_on > 0, c_map * n_on, 0.0)
        L_ref -= np.where(n_off > 0, c_map * n_off, 0.0)

    if not ts:
        return np.empty(0, dtype=np.dtype(
            [("t", "f8"), ("x", "i8"), ("y", "i8"), ("p", "i1")]))
    return events_from_arrays(ts, xs, ys, ps)


def reference_frames(scene, times):
    """Dense true frames at ``times`` -- evaluation only, never fed back."""
    return [(float(t), scene.log_brightness(t)) for t in times]


def sparse_anchors(scene, times, coverage=0.15, noise_sigma=0.02,
                   mask=None, seed=0, corrupt_fraction=0.0, stratified=True):
    """Build sparse anchor frames from the simulator's true frames.

    ``coverage`` is the fraction of pixels that carry an absolute
    measurement in each frame; the rest are masked out and carry no
    information.  With ``stratified=True`` the per-frame masks partition the
    pixels across the anchor sequence so every pixel is supported by some
    anchor while each individual frame stays genuinely sparse.  Anchors are
    independently jittered, and the middle anchor can be corrupted with a
    large offset to exercise anchor-failure handling.
    """
    rng = np.random.RandomState(seed)
    h, w = scene.shape
    n_pixel = h * w
    n_anchor = len(times)

    if stratified and mask is None:
        # Pixels are shuffled, then grouped into chunks of roughly
        # ``coverage`` new pixels each.  Anchor i measures its own chunk
        # plus every earlier chunk: support grows monotonically, so a single
        # failed/corrupt anchor never leaves its pixels unsupported (the
        # next anchor re-covers them), and the final anchor absorbs whatever
        # remains so the whole sensor ends up supported while every frame
        # stays genuinely sparse.
        perm = rng.permutation(n_pixel)
        chunk_size = max(int(round(n_pixel * coverage)), 1)
        chunks = [perm[i:i + chunk_size]
                  for i in range(0, n_pixel, chunk_size)]
        masks = []
        cumulative = np.zeros(n_pixel, dtype=bool)
        for i in range(n_anchor):
            if i < len(chunks):
                cumulative[chunks[i]] = True
            if i == n_anchor - 1:
                for leftover in chunks[n_anchor:]:
                    cumulative[leftover] = True
            masks.append(cumulative.copy().reshape(h, w))
    else:
        masks = None

    anchors = []
    for i, t in enumerate(times):
        logL = scene.log_brightness(t)
        if mask is not None:
            valid = np.asarray(mask, dtype=bool).copy()
        elif masks is not None:
            valid = masks[i]
        else:
            valid = rng.rand(h, w) < coverage
        noisy = logL + rng.randn(h, w) * noise_sigma
        corrupt = corrupt_fraction > 0 and i == int(n_anchor // 2)
        if corrupt:
            noisy = noisy + 3.0  # gross, obviously bad anchor
        anchors.append(make_anchor(float(t), noisy, valid))
    return anchors


def simulate(name="moving_disk", duration=2.0, contrast=0.2, dt=1e-3,
             anchor_every=0.25, anchor_coverage=0.15, seed=0,
             corrupt_fraction=0.0, threshold_jitter=0.0, **scene_kw):
    """End-to-end simulation bundle used by CLI, GUI and tests."""
    scene = SCENES[name](**scene_kw)
    events = generate_events(
        scene, duration, contrast=contrast, dt=dt, seed=seed,
        threshold_jitter=threshold_jitter,
    )
    n_anchor = max(int(round(duration / anchor_every)) + 1, 2)
    anchor_times = np.linspace(0.0, duration, n_anchor)
    anchors = sparse_anchors(
        scene, anchor_times, coverage=anchor_coverage, seed=seed + 100,
        corrupt_fraction=corrupt_fraction,
    )
    return {
        "scene": scene,
        "scene_name": name,
        "events": events,
        "anchors": anchors,
        "anchor_times": anchor_times,
        "contrast": contrast,
        "duration": duration,
        "shape": scene.shape,
    }
