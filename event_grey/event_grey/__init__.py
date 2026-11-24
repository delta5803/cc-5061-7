"""Event-camera log-brightness reconstruction toolkit.

The package reconstructs high-temporal-resolution grayscale frames from an
event stream (pixel coordinates, polarity, timestamps) plus a small number of
sparse absolute-brightness anchor frames.  Events describe *changes* of log
brightness; without an anchor or an explicit prior there is no information
about the absolute level of a pixel, and this package never fabricates one
from event counts.
"""

from .model import (
    EVENT_DTYPE,
    NEUTRAL_GREY,
    sort_events,
    events_from_arrays,
    make_anchor,
    make_prior,
)
from .reconstruct import Reconstructor, ReconstructionResult

__all__ = [
    "EVENT_DTYPE",
    "NEUTRAL_GREY",
    "sort_events",
    "events_from_arrays",
    "make_anchor",
    "make_prior",
    "Reconstructor",
    "ReconstructionResult",
]
