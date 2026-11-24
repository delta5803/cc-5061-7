"""Core data structures and event-stream semantics.

An event is the threshold crossing of a single pixel's log brightness::

    L(x, y, t) = ln(I(x, y, t) + eps)

A positive (``p=1``) event means the log brightness increased by the pixel's
contrast threshold ``C`` since the last emitted event; a negative event
(``p=0``) means it decreased by ``C``.  Events therefore carry only signed
*increments* of log brightness.  They never carry an absolute level.
"""

from __future__ import annotations

import numpy as np

# Structured array layout used for every event stream in the package.
# ``p`` is 0 for an OFF (darker) event and 1 for an ON (brighter) event.
EVENT_DTYPE = np.dtype(
    [("t", np.float64), ("x", np.int64), ("y", np.int64), ("p", np.int8)]
)

# Epsilon used whenever a log brightness is taken of an intensity image.
LOG_EPS = 1e-3

# Explicit prior used for pixels that never receive any support.  It is a
# *neutral mid-grey* prior, never derived from event counts, and every pixel
# carrying it keeps ``supported == False`` so downstream code and the UI can
# flag it as unsupported rather than presenting it as measured brightness.
NEUTRAL_GREY = float(np.log(0.5 + LOG_EPS))


def events_from_arrays(t, x, y, p):
    """Build a canonical event array from column arrays.

    Polarity values of {-1, +1} are normalized to {0, 1}.
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.int64)
    if not (t.shape == x.shape == y.shape == p.shape):
        raise ValueError("t, x, y, p must have identical shapes")
    if np.any((p != 0) & (p != 1) & (p != -1)):
        raise ValueError("polarity must be 0/1 (or -1 for OFF)")
    p = np.where(p == -1, 0, p).astype(np.int8)
    events = np.empty(t.shape[0], dtype=EVENT_DTYPE)
    events["t"] = t
    events["x"] = x
    events["y"] = y
    events["p"] = p
    return events


def sort_events(events):
    """Canonicalize an event stream.

    Ordering is lexicographic on ``(t, y, x, p)`` so that:

    * arbitrary input order is handled deterministically;
    * several events sharing a timestamp are processed in a fixed,
      reproducible spatial order rather than whatever order they arrived in;
    * exact duplicate events are identifiable and removed (the same pixel
      cannot cross the threshold twice at the same instant).
    """
    events = np.asarray(events, dtype=EVENT_DTYPE)
    order = np.lexsort((events["p"], events["x"], events["y"], events["t"]))
    events = events[order]
    if events.size:
        dup = (
            (events["t"][1:] == events["t"][:-1])
            & (events["x"][1:] == events["x"][:-1])
            & (events["y"][1:] == events["y"][:-1])
            & (events["p"][1:] == events["p"][:-1])
        )
        if np.any(dup):
            keep = np.concatenate(([True], ~dup))
            events = events[keep]
    return events


def make_anchor(t, logL, valid=None, weight=1.0):
    """Build an anchor-frame record (log brightness + support mask).

    ``logL`` is a 2-D array of log brightness.  ``valid`` is the support
    mask: pixels that are False carry no absolute information at this anchor
    and must not be treated as measurements.  A full frame passes no mask.
    """
    logL = np.asarray(logL, dtype=np.float64)
    if logL.ndim != 2:
        raise ValueError("anchor logL must be a 2-D array")
    if valid is None:
        valid = np.ones(logL.shape, dtype=bool)
    else:
        valid = np.array(valid, dtype=bool, copy=True)
        if valid.shape != logL.shape:
            raise ValueError("valid mask shape must match logL")
    return {
        "t": float(t),
        "logL": np.array(logL, copy=True),
        "valid": valid,
        "weight": float(weight),
    }


def make_prior(shape, logL=None):
    """Build the explicit per-pixel prior used before the first anchor.

    The default is a uniform neutral mid-grey prior and *no* support, i.e.
    the reconstruction can only say "unknown absolute level" until an anchor
    (or a caller-provided supported prior) arrives.
    """
    h, w = int(shape[0]), int(shape[1])
    if logL is None:
        logL = np.full((h, w), NEUTRAL_GREY, dtype=np.float64)
        supported = np.zeros((h, w), dtype=bool)
    else:
        logL = np.asarray(logL, dtype=np.float64)
        if logL.shape != (h, w):
            raise ValueError("prior logL shape mismatch")
        supported = np.ones((h, w), dtype=bool)
    return {"logL": np.array(logL, dtype=np.float64, copy=True),
            "supported": supported}
