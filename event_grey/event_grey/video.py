"""Render reconstructed grayscale frames to a real video file.

The output is a video of the reconstructed *grayscale field* (optionally
side-by-side with event distribution and reference view), not an animated
scatter of events.  Encoding uses the ffmpeg binary bundled with
``imageio-ffmpeg`` so no system installation is required.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
import imageio.v2 as imageio

from .model import LOG_EPS


def logL_to_uint8(logL, lo=None, hi=None):
    """Map log brightness to display grayscale with a fixed shared scale."""
    if lo is None:
        lo = float(np.log(0.01 + LOG_EPS))
    if hi is None:
        hi = float(np.log(1.0 + LOG_EPS))
    img = (np.clip((logL - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    return img


def event_window_image(events, t0, t1, shape):
    """2-D event distribution image over [t0, t1): red ON, blue OFF."""
    h, w = shape
    m = (events["t"] >= t0) & (events["t"] < t1)
    ev = events[m]
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    if ev.size:
        on = ev["p"] == 1
        for mask, color in ((on, (220, 60, 60)), (~on, (60, 90, 220))):
            sel = ev[mask]
            if sel.size:
                canvas[sel["y"], sel["x"]] = color
    return canvas


def unsupported_overlay(grey, supported, alpha=0.35):
    """Blend a light amber tint over unsupported pixels while keeping the
    underlying grayscale dominant, so missing support is readable without
    hiding the field itself."""
    rgb = np.repeat(grey[:, :, None], 3, axis=2).astype(np.float32)
    marker = np.array([255, 200, 40], dtype=np.float32)
    unsup = ~supported
    rgb[unsup] = (1 - alpha) * rgb[unsup] + alpha * marker
    return rgb.astype(np.uint8)


def write_video(path, result, scene=None, show_events=True, show_reference=False,
                fps=20, events_window=0.05):
    """Write an MP4 of the reconstruction.

    Panels (when enabled): left = reconstructed grayscale with unsupported
    pixels highlighted, middle = event distribution in the time window,
    right = simulator reference (evaluation only, clearly labelled).
    """
    frames = result.frames
    shape = result.shape
    writer = imageio.get_writer(
        path, fps=fps, codec="libx264", quality=7, macro_block_size=None
    )
    try:
        for i, (t, logL, supported) in enumerate(frames):
            grey = logL_to_uint8(logL)
            left = unsupported_overlay(grey, supported)
            panels = [left]
            if show_events:
                t1 = t
                t0 = t - events_window
                panels.append(event_window_image(
                    result.events, t0, t1, shape))
            if show_reference and scene is not None:
                ref_grey = logL_to_uint8(scene.log_brightness(t))
                panels.append(np.repeat(ref_grey[:, :, None], 3, axis=2))
            frame = np.concatenate(panels, axis=1)
            writer.append_data(frame)
    finally:
        writer.close()
    return path


def write_frame_image(path, logL, supported=None):
    grey = logL_to_uint8(logL)
    if supported is not None and not supported.all():
        grey = unsupported_overlay(grey, supported)
    Image.fromarray(grey).save(path)
    return path
