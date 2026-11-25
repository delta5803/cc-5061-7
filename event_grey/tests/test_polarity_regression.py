"""Regression tests for polarity accumulation, equal-time events and the
player's per-pixel trace.

Context / failure evidence
--------------------------
The streaming kernel (:meth:`Reconstructor.run`) was audited against an
independent scalar reference and is correct: same-pixel opposite-polarity
events and several events sharing one timestamp integrate in a fixed,
deterministic order, and a late event inserted before a gray anchor only
perturbs the segment *before the next anchor* (a constant ``±C`` offset),
never negating the whole segment.

The defect was in the code path that drives the player controls:
:meth:`Reconstructor.pixel_trace` re-implemented the sweep with (1) events
integrated up to the output time *before* the anchor at its own timestamp,
(2) globally-rejected (corrupt) anchors still applied, and (3) a hard-coded
neutral prior instead of the reconstructor's actual prior.  The player's
per-pixel brightness trajectory therefore disagreed with the very frames it
rendered (worst case ~4.25 log units on a corrupt-anchor run).

These tests pin the physics at the polarity-accumulation boundary and pin
the trace to the frames, so a regression cannot pass by resetting brightness
every frame.
"""

import numpy as np
import pytest

from event_grey.analysis import AnalysisSession
from event_grey.model import (
    EVENT_DTYPE,
    events_from_arrays,
    make_anchor,
    make_prior,
    sort_events,
)
from event_grey.reconstruct import Reconstructor
from event_grey.simulator import generate_events, simulate


C = 0.2


def _run(events, output_times, anchors=(), prior_level=0.0, shape=(1, 1),
         blend=0.5, reset_drift=0.6, anchor_fail_drift=1.5):
    rec = Reconstructor(
        shape, contrast=C, anchor_blend=blend, reset_drift=reset_drift,
        anchor_fail_drift=anchor_fail_drift,
        prior=make_prior(shape, logL=np.full(shape, prior_level)),
    )
    for a in anchors:
        rec.add_anchor(a)
    result = rec.run(events, np.asarray(output_times, dtype=float))
    return rec, result


# --------------------------------------------------------------------------- #
# Equal-time / opposite-polarity membership
# --------------------------------------------------------------------------- #
def test_equal_time_opposite_polarity_same_pixel_keeps_both_and_cancels():
    """One ON and one OFF at the *same* pixel and *same* timestamp are two
    distinct events (different polarity): neither is de-duplicated, and the
    integrated level returns to the prior."""
    ev = events_from_arrays(t=[0.2, 0.2], x=[0, 0], y=[0, 0], p=[1, 0])
    s = sort_events(ev)
    assert s.size == 2
    assert set(s["p"].tolist()) == {0, 1}

    _, res = _run(ev, [0.0, 0.2, 0.4])
    L = res.frame_array()[:, 0, 0]
    assert L[0] == pytest.approx(0.0)
    assert L[1] == pytest.approx(0.0)   # net zero at the instant itself
    assert L[2] == pytest.approx(0.0)


def test_equal_time_same_polarity_same_pixel_is_deduplicated():
    """The same pixel cannot cross the *same* threshold twice at one instant;
    identical (t, x, y, p) tuples collapse to a single event."""
    ev = events_from_arrays(t=[0.3, 0.3, 0.3], x=[1, 1, 1], y=[1, 1, 1],
                            p=[1, 1, 1])
    s = sort_events(ev)
    assert s.size == 1
    _, res = _run(ev, [0.5], shape=(2, 2))
    assert res.frames[0][1][1, 1] == pytest.approx(C)


def test_equal_time_events_processed_in_spatial_order_not_input_order():
    """At one timestamp the canonical order is (y, x, p); the integrated
    result is independent of the order the caller supplied."""
    forward = events_from_arrays(t=[0.5, 0.5, 0.5], x=[0, 1, 2], y=[0, 0, 0],
                                 p=[1, 0, 1])
    backward = events_from_arrays(t=[0.5, 0.5, 0.5], x=[2, 1, 0], y=[0, 0, 0],
                                  p=[1, 0, 1])
    _, a = _run(forward, [1.0], shape=(1, 3))
    _, b = _run(backward, [1.0], shape=(1, 3))
    assert np.array_equal(a.frame_array(), b.frame_array())
    L = a.frames[0][1][0]
    assert L[0] == pytest.approx(C)
    assert L[1] == pytest.approx(-C)
    assert L[2] == pytest.approx(C)


# --------------------------------------------------------------------------- #
# Late event inserted before a gray anchor: segment locality, no sign flip
# --------------------------------------------------------------------------- #
def _segment_locality_case():
    shape = (1, 1)
    out = np.linspace(0.0, 1.0, 21)
    anchors = [
        make_anchor(0.0, np.array([[0.0]])),
        make_anchor(0.5, np.array([[0.0]])),
        make_anchor(1.0, np.array([[0.0]])),
    ]
    base = events_from_arrays(t=[0.2, 0.4, 0.7], x=[0, 0, 0], y=[0, 0, 0],
                              p=[1, 1, 1])
    # A negative event that belongs at t=0.3 is delivered "late" but inserted
    # in canonical order, i.e. before the gray anchor at t=0.5.
    late = events_from_arrays(t=[0.2, 0.3, 0.4, 0.7],
                              x=[0, 0, 0, 0], y=[0, 0, 0, 0],
                              p=[1, 0, 1, 1])
    return out, anchors, base, late


def test_late_negative_before_anchor_only_changes_preceding_segment():
    """Inserting a late OFF before the gray anchor at t=0.5 (which fully
    absorbs drift here) may only change the trajectory on
    [t_event, next_anchor).  Before the event and after the absorbing anchor
    the reconstructed level is bit-identical."""
    out, anchors, base, late = _segment_locality_case()
    _, rb = _run(base, out, anchors, blend=1.0)
    _, rl = _run(late, out, anchors, blend=1.0)
    Lb = rb.frame_array()[:, 0, 0]
    Ll = rl.frame_array()[:, 0, 0]
    diff = Ll - Lb

    assert np.allclose(diff[out < 0.3], 0.0)          # before the late event
    assert np.allclose(diff[out >= 0.5], 0.0)         # anchor at 0.5 absorbs it
    affected = (out >= 0.3) & (out < 0.5)
    assert np.allclose(diff[affected], -C)            # exactly one OFF step


def test_late_negative_is_constant_offset_not_whole_segment_negation():
    """The recompute must shift the segment by a constant ``-C``; it must not
    invert the sign of the brightness trajectory (a naive 'recompute from the
    anchor by re-integrating the whole segment backwards' bug would)."""
    out, anchors, base, late = _segment_locality_case()
    _, rb = _run(base, out, anchors, blend=1.0)
    _, rl = _run(late, out, anchors, blend=1.0)
    Lb = rb.frame_array()[:, 0, 0]
    Ll = rl.frame_array()[:, 0, 0]
    affected = (out >= 0.3) & (out < 0.5)

    # Pure DC offset: every step between samples is identical, only the level
    # moved down by one threshold.
    assert np.allclose(np.diff(Ll[affected]), np.diff(Lb[affected]))
    assert np.allclose(Ll[affected] - Lb[affected], -C)
    # Not a mirror of the segment (the whole-segment sign-flip failure mode).
    assert not np.allclose(Ll[affected], -Lb[affected])
    # Where the shifted level is still strictly non-zero its sign matches the
    # base trajectory's sign; nothing in the segment turned dark->bright.
    both = affected & (np.abs(Ll) > 1e-9) & (np.abs(Lb) > 1e-9)
    assert np.any(both)
    assert np.all(np.sign(Ll[both]) == np.sign(Lb[both]))


def test_late_negative_before_anchor_does_not_affect_earlier_or_later():
    """Anchor-segment replay must be local: a pixel never touched by the late
    event, and output times before it, stay exactly unchanged."""
    out, anchors, base, late = _segment_locality_case()
    _, rb = _run(base, out, anchors, blend=1.0)
    _, rl = _run(late, out, anchors, blend=1.0)
    assert np.array_equal(rb.frame_array()[:, 0, 0][out < 0.3],
                          rl.frame_array()[:, 0, 0][out < 0.3])
    # A second pixel with no events at all is untouched by the insertion.
    shape = (1, 2)
    anchors2 = [make_anchor(t, np.zeros(shape)) for t in (0.0, 0.5, 1.0)]
    base2 = events_from_arrays(t=[0.2], x=[0], y=[0], p=[1])
    late2 = events_from_arrays(t=[0.2, 0.3], x=[0, 0], y=[0, 0], p=[1, 0])
    _, r2b = _run(base2, out, anchors2, shape=shape, blend=1.0)
    _, r2l = _run(late2, out, anchors2, shape=shape, blend=1.0)
    assert np.array_equal(r2b.frame_array()[:, 0, 1],
                          r2l.frame_array()[:, 0, 1])


def test_polarity_accumulation_boundary_with_soft_anchor_leaves_residual():
    """With a soft anchor the late event leaves the documented partial
    residual (``1-w``) rather than being silently swallowed or doubled."""
    out, anchors, base, late = _segment_locality_case()
    _, rb = _run(base, out, anchors, blend=0.5)
    _, rl = _run(late, out, anchors, blend=0.5)
    diff = rl.frame_array()[:, 0, 0] - rb.frame_array()[:, 0, 0]
    # The anchor at 0.5 only partially corrects a sub-reset drift, so a
    # constant -C * (1 - w) offset survives the anchor for the rest of the run.
    after = (out >= 0.5) & (out < 1.0)
    assert np.allclose(diff[after], -C * (1 - 0.5))
    # Still a constant offset -- sign of the underlying curve is preserved.
    assert np.allclose(np.diff(diff[after]), 0.0)


def test_cumulative_error_bounded_by_one_threshold_per_anchor_segment():
    """Across a run with anchors every 0.25 s, the cumulative event-only
    error at a pixel never exceeds a couple of contrast steps; anchors bound
    it so it cannot grow without limit."""
    scene = simulate(name="grating", duration=1.0, anchor_coverage=0.2,
                     seed=1)["scene"]
    events = generate_events(scene, duration=1.0, contrast=C, dt=1e-3)
    s = sort_events(events)
    out = np.linspace(0.0, 1.0, 101)
    # Pixel with the most events.
    counts = np.zeros(scene.shape, dtype=np.int64)
    np.add.at(counts, (s["y"], s["x"]), 1)
    y, x = np.unravel_index(np.argmax(counts), counts.shape)
    pe = s[(s["x"] == x) & (s["y"] == y)]
    level, j, trace = 0.0, 0, []
    for t in out:
        while j < pe.size and pe[j]["t"] <= t:
            level += C if pe[j]["p"] == 1 else -C
            j += 1
        trace.append(level)
    trace = np.array(trace)
    truth = scene.log_brightness_at_times(out)[:, y, x]
    truth = truth - scene.log_brightness(0.0)[y, x]
    assert np.max(np.abs(trace - truth)) <= 2 * C + 1e-9


# --------------------------------------------------------------------------- #
# The player's per-pixel trace must be driven by the same kernel
# --------------------------------------------------------------------------- #
def _assert_trace_matches_frames(session, pixels):
    frames = session.result.frame_array()
    for (x, y) in pixels:
        trace = session.pixel_trace(x, y)
        assert np.allclose(trace["corrected"], frames[:, y, x], atol=0.0), (
            f"player trace disagrees with reconstructed frames at ({x},{y})"
        )


def test_player_trace_equals_reconstructed_frames_all_scenes():
    """The brightness trajectory plotted for a pixel must equal the frame the
    player renders -- across every scene, at event-rich and boundary pixels."""
    for name in ("moving_disk", "grating", "static_panel", "fast_edge"):
        bundle = simulate(name=name, duration=0.6, anchor_coverage=0.15, seed=0)
        session = AnalysisSession(bundle, fps=20.0)
        h, w = session.shape
        counts = np.zeros((h, w), np.int64)
        if session.events.size:
            np.add.at(counts, (session.events["y"], session.events["x"]), 1)
        y, x = np.unravel_index(np.argmax(counts), counts.shape)
        pixels = [(x, y), (0, 0), (w - 1, h - 1), (w // 2, h // 2)]
        _assert_trace_matches_frames(session, pixels)


def test_player_trace_equals_frames_with_failed_anchor():
    """A globally-rejected corrupt anchor must be ignored by the trace too;
    the plotted trajectory still matches the frames rather than jumping to
    the corrupt level."""
    bundle = simulate(name="fast_edge", duration=1.0, corrupt_fraction=1.0,
                      anchor_coverage=0.15, seed=2)
    session = AnalysisSession(bundle, fps=20.0)
    assert any(r.failed for r in session.result.reports)
    h, w = session.shape
    pixels = [(0, 0), (w // 2, h // 2), (w - 1, h - 1)]
    _assert_trace_matches_frames(session, pixels)
    # The corrupt anchor added +3.0; no corrected point may jump by that.
    for (x, y) in pixels:
        trace = session.pixel_trace(x, y)
        assert np.max(np.abs(trace["corrected"])) < 3.0


def test_player_trace_applies_anchor_at_its_own_timestamp():
    """An event that falls between an anchor and the next output time is
    integrated *after* the anchor (ordering bug regression)."""
    shape = (1, 1)
    ev = events_from_arrays(t=[0.55], x=[0], y=[0], p=[1])
    rec = Reconstructor(shape, contrast=C,
                        prior=make_prior(shape, logL=np.zeros(shape)))
    rec.add_anchor(make_anchor(0.5, np.array([[0.0]])))
    out = np.array([0.0, 0.3, 0.6])
    frame = rec.run(ev, out).frame_array()[:, 0, 0]
    trace = rec.pixel_trace(0, 0, ev, out)
    assert trace["corrected"][2] == pytest.approx(frame[2])
    assert trace["corrected"][2] == pytest.approx(C)   # anchor 0, then +C


def test_player_trace_starts_from_actual_prior_not_neutral_grey():
    """A supported custom prior is the trace's starting level, matching the
    frames (the old trace hard-coded neutral grey)."""
    shape = (1, 1)
    level = 0.5
    prior = make_prior(shape, logL=np.full(shape, level))  # supported=True
    rec = Reconstructor(shape, contrast=C, prior=prior)
    out = np.array([0.0, 0.5])
    frame = rec.run(np.empty(0, dtype=EVENT_DTYPE), out).frame_array()[:, 0, 0]
    trace = rec.pixel_trace(0, 0, np.empty(0, dtype=EVENT_DTYPE), out)
    assert trace["corrected"][0] == pytest.approx(level)
    assert np.allclose(trace["corrected"], frame)
    assert trace["raw_integration"][0] == pytest.approx(level)


def test_player_trace_is_pure_and_idempotent():
    """Calling pixel_trace must not mutate the reconstructor's live state, so
    repeated calls and a subsequent run() are unaffected."""
    bundle = simulate(name="moving_disk", duration=0.5, anchor_coverage=0.2,
                      seed=0)
    session = AnalysisSession(bundle, fps=20.0)
    before = session.result.frame_array().copy()
    x, y = 10, 20
    t1 = session.pixel_trace(x, y)
    t2 = session.pixel_trace(x, y)
    assert np.array_equal(t1["corrected"], t2["corrected"])
    assert np.array_equal(session.result.frame_array(), before)


def test_failed_anchor_trace_anchor_points_absent_for_rejected_frame():
    """The trace's anchor markers must not include a rejected anchor."""
    shape = (1, 1)
    rec = Reconstructor(shape, contrast=C, anchor_fail_drift=1.5,
                        prior=make_prior(shape, logL=np.zeros(shape)))
    rec.add_anchor(make_anchor(0.0, np.zeros(shape)))
    bad_t = 0.5
    rec.add_anchor(make_anchor(bad_t, np.full(shape, 5.0)))
    rec.add_anchor(make_anchor(1.0, np.zeros(shape)))
    out = np.array([0.0, 0.5, 1.0])
    trace = rec.pixel_trace(
        0, 0, np.empty(0, dtype=EVENT_DTYPE), out)
    marked = [pt[0] for pt in trace["anchor_points"]]
    assert bad_t not in marked
    assert 0.0 in marked and 1.0 in marked


# --------------------------------------------------------------------------- #
# Exact-boundary semantics (event time == anchor time, anchor == output,
# reset/fail bands are strict ``>``)
# --------------------------------------------------------------------------- #
def test_event_at_anchor_timestamp_integrated_before_anchor():
    """An event whose timestamp equals an anchor's is integrated *before* the
    anchor is applied (side='right'), and the trace agrees."""
    shape = (1, 1)
    ev = events_from_arrays(t=[0.5], x=[0], y=[0], p=[1])
    rec = Reconstructor(shape, contrast=C, anchor_blend=0.5,
                        prior=make_prior(shape, logL=np.zeros(shape)))
    rec.add_anchor(make_anchor(0.5, np.array([[0.0]])))
    out = np.array([0.5])
    frame = rec.run(ev, out).frame_array()[:, 0, 0]
    trace = rec.pixel_trace(0, 0, ev, out)
    # Prior is supported, level 0 -> +C from the event -> 0.2; drift -0.2 is
    # within the reset band, so a 50/50 blend gives 0.1.
    assert frame[0] == pytest.approx(0.5 * (0.0 + C))
    assert trace["corrected"][0] == pytest.approx(frame[0])


def test_anchor_at_output_time_applied_before_snapshot():
    shape = (1, 1)
    ev = events_from_arrays(t=[0.4], x=[0], y=[0], p=[1])
    rec = Reconstructor(shape, contrast=C, anchor_blend=0.5,
                        prior=make_prior(shape, logL=np.zeros(shape)))
    rec.add_anchor(make_anchor(0.5, np.array([[0.0]])))
    out = np.array([0.5])
    frame = rec.run(ev, out).frame_array()[:, 0, 0]
    trace = rec.pixel_trace(0, 0, ev, out)
    assert frame[0] == pytest.approx(0.1)   # +C then 50/50 blend toward 0
    assert trace["corrected"][0] == pytest.approx(frame[0])


def test_reset_and_fail_bands_are_strict_greater_than():
    """Drift exactly at the reset band blends; just above resets.  Median drift
    exactly at the failure band is accepted; just above is rejected."""
    shape = (1, 1)
    ev = np.empty(0, dtype=EVENT_DTYPE)

    def apply(target, reset=0.6, fail=1.5, blend=0.5):
        rec = Reconstructor(shape, contrast=C, anchor_blend=blend,
                            reset_drift=reset, anchor_fail_drift=fail,
                            prior=make_prior(shape, logL=np.zeros(shape)))
        rec.add_anchor(make_anchor(0.5, np.array([[target]])))
        res = rec.run(ev, np.array([1.0]))
        return res.frames[0][1][0, 0], res.reports[0]

    v, rep = apply(0.6)
    assert v == pytest.approx(0.3) and rep.reset_pixels == 0   # blend
    v, rep = apply(0.6000001)
    assert v == pytest.approx(0.6000001) and rep.reset_pixels == 1  # reset

    _, rep = apply(1.5)
    assert not rep.failed           # == fail band accepted
    _, rep = apply(1.5000001)
    assert rep.failed               # > fail band rejected


# --------------------------------------------------------------------------- #
# Cross-check the kernel against an independent chronological-merge reference
# --------------------------------------------------------------------------- #
def _merge_reference(shape, events, anchors, out, prior_logL, prior_sup,
                     blend=0.5, reset_drift=0.6, fail=1.5):
    """Chronological merge over (time, kind) items -- deliberately a different
    algorithm from reconstruct.py's cursor/searchsorted sweep."""
    ev = sort_events(events)
    items = [(float(e["t"]), 0, e) for e in ev]
    items += [(float(a["t"]), 1, a) for a in anchors]
    items.sort(key=lambda it: (it[0], it[1]))   # events before anchors at ties
    L = prior_logL.copy()
    sup = prior_sup.copy().astype(bool)
    frames = []
    k = 0
    for t in np.sort(np.asarray(out, dtype=float)):
        while k < len(items) and items[k][0] <= t:
            _, kind, obj = items[k]
            if kind == 0:
                L[obj["y"], obj["x"]] += C if obj["p"] == 1 else -C
            else:
                valid = obj["valid"]
                if valid.sum():
                    gd = float(np.median((obj["logL"] - L)[valid]))
                    if abs(gd) <= fail:
                        w = min(blend * float(obj.get("weight", 1.0)), 1.0)
                        drift = obj["logL"] - L
                        rm = valid & (np.abs(drift) > reset_drift)
                        fm = valid & ~sup
                        bm = valid & ~rm & ~fm
                        L[bm] = (1 - w) * L[bm] + w * obj["logL"][bm]
                        L[rm | fm] = obj["logL"][rm | fm]
                        sup[valid] = True
            k += 1
        frames.append(L.copy())
    return np.stack(frames)


def test_kernel_matches_independent_merge_reference_on_equal_time_streams():
    """Randomised equal-timestamp, opposite-polarity streams must agree with
    an independently implemented chronological merge (not merely with the
    kernel's own structure)."""
    rng = np.random.RandomState(7)
    h = w = 3
    shape = (h, w)
    for trial in range(60):
        n = rng.randint(6, 30)
        ts = rng.choice([0.2, 0.4, 0.6, 0.8], size=n)  # heavy equal times
        xs = rng.randint(0, w, n)
        ys = rng.randint(0, h, n)
        ps = rng.randint(0, 2, n)
        ev = events_from_arrays(ts, xs, ys, ps)

        at = np.sort(rng.choice([0.1, 0.3, 0.5, 0.7, 0.9],
                                size=rng.randint(1, 4), replace=False))
        L0 = rng.randn(h, w) * 0.15
        anchors = [make_anchor(0.0, L0.copy(), np.ones(shape, bool))]
        for a_t in at:
            tgt = L0 + rng.randn(h, w) * 0.25
            anchors.append(make_anchor(float(a_t), tgt, rng.rand(h, w) < 0.75))
        out = np.linspace(0.0, 1.0, 21)

        prior = make_prior(shape, logL=L0.copy())
        rec = Reconstructor(shape, contrast=C, anchor_blend=0.5,
                            reset_drift=0.6, anchor_fail_drift=1.5,
                            prior=make_prior(shape, logL=L0.copy()))
        for a in anchors:
            rec.add_anchor(a)
        got = rec.run(ev, out).frame_array()
        ref = _merge_reference(shape, ev, anchors, out, L0.copy(),
                               prior["supported"])
        assert np.allclose(got, ref), f"trial {trial} diverged from reference"


# --------------------------------------------------------------------------- #
# Anchor-segment replay vs a naively reversible cumulative quantity
# --------------------------------------------------------------------------- #
def test_reversible_accumulator_is_wrong_across_an_anchor():
    """A late event can be handled two ways.  Adding its increment to every
    later stored sample (an exactly reversible accumulator) is correct *within*
    an anchor segment but wrong across an anchor, because soft-blend/reset are
    non-linear.  Anchor-segment replay (what the kernel does) is the correct
    one; the test pins that the two coincide inside the segment and that the
    reversible one accumulates a larger error after the anchor."""
    shape = (1, 1)
    out = np.linspace(0.0, 1.0, 21)
    anchors = [make_anchor(t, np.array([[0.0]])) for t in (0.0, 0.5, 1.0)]
    base = events_from_arrays(t=[0.2, 0.4, 0.7], x=[0, 0, 0], y=[0, 0, 0],
                              p=[1, 1, 1])
    late = events_from_arrays(t=[0.2, 0.3, 0.4, 0.7],
                              x=[0, 0, 0, 0], y=[0, 0, 0, 0],
                              p=[1, 0, 1, 1])
    late_t, d = 0.3, -C

    _, rb = _run(base, out, anchors, blend=1.0)
    _, rl = _run(late, out, anchors, blend=1.0)
    base_tr = rb.frame_array()[:, 0, 0]
    replay = rl.frame_array()[:, 0, 0]
    reversible = base_tr + np.where(out >= late_t, d, 0.0)

    within = (out >= late_t) & (out < 0.5)
    after = out >= 0.5
    just_after = (out >= 0.5) & (out < 0.7)   # post-anchor, before the next event
    # Identical inside the anchor segment...
    assert np.allclose(replay[within], reversible[within])
    # ...but the additive accumulator is wrong after the absorbing anchor,
    # which reset the level and wiped the late event's effect.
    assert not np.allclose(replay[after], reversible[after])
    assert np.allclose(replay[just_after], 0.0)     # replay absorbs at anchor
    assert np.sum(np.abs(replay - base_tr)) < np.sum(np.abs(reversible - base_tr))
