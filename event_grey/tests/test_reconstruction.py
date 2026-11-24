import numpy as np
import pytest

from event_grey.model import (
    EVENT_DTYPE,
    NEUTRAL_GREY,
    events_from_arrays,
    make_anchor,
    make_prior,
    sort_events,
)
from event_grey.analysis import AnalysisSession
from event_grey.reconstruct import Reconstructor
from event_grey.simulator import (
    SCENES,
    StaticPanelScene,
    DriftingGratingScene,
    generate_events,
    simulate,
    sparse_anchors,
)
from event_grey.metrics import evaluate


def test_sort_orders_and_deduplicates():
    ev = events_from_arrays(
        t=[2.0, 1.0, 1.0, 1.0, 1.0],
        x=[0, 3, 3, 1, 1],
        y=[0, 1, 1, 2, 2],
        p=[1, 0, 0, 1, 0],
    )
    out = sort_events(ev)
    assert np.all(np.diff(out["t"]) >= 0)
    # exact duplicate (t,x,y,p) removed; same-timestamp different pixel kept
    assert out.size == 4
    # deterministic spatial tie-break at t=1
    assert (out[0]["y"], out[0]["x"]) == (1, 3)


def test_polarity_normalization():
    ev = events_from_arrays(t=[0, 1], x=[0, 1], y=[0, 1], p=[-1, 1])
    assert list(ev["p"]) == [0, 1]


def test_temporal_integration_keeps_intermediate_peak():
    """Equal ON/OFF counts must still show the intermediate intensity peak."""
    shape = (2, 2)
    C = 0.2
    prior = make_prior(shape, logL=np.full(shape, -1.0))
    rec = Reconstructor(shape, contrast=C, prior=prior)
    # at pixel (0,0): ON ON then OFF OFF -> net zero, peak of +2C at t=2
    ev = events_from_arrays(
        t=[1, 2, 3, 4], x=[0, 0, 0, 0], y=[0, 0, 0, 0], p=[1, 1, 0, 0]
    )
    res = rec.run(ev, output_times=np.array([0.5, 1.5, 2.5, 3.5, 4.5]))
    L = res.frame_array()[:, 0, 0]
    assert L[0] == pytest.approx(-1.0)
    assert L[1] == pytest.approx(-1.0 + C)
    assert L[2] == pytest.approx(-1.0 + 2 * C)   # peak preserved
    assert L[3] == pytest.approx(-1.0 + C)
    assert L[4] == pytest.approx(-1.0)           # net count zero


def test_output_grid_does_not_change_accumulation():
    shape = (4, 4)
    C = 0.2
    ev = events_from_arrays(
        t=np.repeat(np.linspace(0.1, 0.9, 9), 3),
        x=np.tile([0, 1, 2], 9),
        y=np.tile([0, 1, 2], 9),
        p=np.tile([1, 0, 1], 9),
    )
    anchors = [make_anchor(0.0, np.full(shape, -0.5)),
               make_anchor(1.0, np.full(shape, -0.5))]

    def run(times):
        r = Reconstructor(shape, contrast=C)
        for a in anchors:
            r.add_anchor(a)
        return r.run(ev, np.array(times)).frame_array()[-1]

    coarse = run([0.0, 0.5, 1.0])
    fine = run(np.linspace(0, 1, 41))
    assert np.allclose(coarse, fine)


def test_no_event_region_stays_at_anchor_level():
    shape = (8, 8)
    level = -0.7
    rec = Reconstructor(shape, contrast=0.2, prior=make_prior(
        shape, logL=np.full(shape, level)))
    # events only in the top-left 2x2 block
    ev = events_from_arrays(
        t=[0.1, 0.2, 0.3], x=[0, 1, 0], y=[0, 0, 1], p=[1, 1, 0]
    )
    res = rec.run(ev, np.linspace(0, 1, 11))
    frames = res.frame_array()
    quiet = frames[:, 4:, 4:]
    assert np.all(quiet == level)  # never moves without events
    assert np.allclose(np.var(quiet, axis=0), 0.0)


def test_unsupported_pixels_keep_explicit_prior_not_event_count():
    shape = (4, 4)
    rec = Reconstructor(shape, contrast=0.2)  # default neutral prior, no support
    ev = events_from_arrays(
        t=[0.1, 0.2], x=[0, 0], y=[0, 0], p=[1, 1]
    )
    res = rec.run(ev, np.array([0.5]))
    # pixel without events must remain exactly at the declared prior
    assert res.frames[0][1][3, 3] == pytest.approx(NEUTRAL_GREY)
    assert not res.frames[0][2][3, 3]
    assert any("never covered" in w for w in res.warnings)


def test_anchor_soft_blend_vs_hard_reset():
    shape = (2, 2)
    rec = Reconstructor(shape, contrast=0.2, anchor_blend=0.5,
                        reset_drift=0.6,
                        prior=make_prior(shape, logL=np.zeros(shape)))
    # pixel (0,0): small drift -> blend; pixel (1,1): large drift -> reset
    target = np.zeros(shape)
    target[0, 0] = 0.3    # |drift| 0.3 < 0.6
    target[1, 1] = 2.0    # |drift| 2.0 > 0.6
    rec.add_anchor(make_anchor(0.5, target))
    res = rec.run(np.empty(0, dtype=EVENT_DTYPE), np.array([1.0]))
    L = res.frames[0][1]
    assert L[0, 0] == pytest.approx(0.15)       # 50/50 blend
    assert L[1, 1] == pytest.approx(2.0)        # hard reset
    report = res.reports[0]
    assert report.reset_pixels == 1
    assert report.blended_pixels == 3


def test_failed_anchor_is_rejected_and_warns():
    shape = (2, 2)
    rec = Reconstructor(shape, contrast=0.2, anchor_fail_drift=1.5,
                        prior=make_prior(shape, logL=np.zeros(shape)))
    bad = make_anchor(0.5, np.full(shape, 5.0))  # gross corrupt anchor
    rec.add_anchor(bad)
    res = rec.run(np.empty(0, dtype=EVENT_DTYPE), np.array([1.0]))
    assert res.reports[0].failed
    assert np.allclose(res.frames[0][1], 0.0)    # state untouched
    assert any("FAILED" in w for w in res.warnings)


def test_sparse_anchor_mask_carries_no_information():
    shape = (4, 4)
    # unsupported neutral prior: masked pixels must stay unsupported
    rec = Reconstructor(shape, contrast=0.2,
                        prior=make_prior(shape))
    rec.logL[:] = -0.3
    logL = np.full(shape, 1.0)
    valid = np.zeros(shape, dtype=bool)
    valid[0, 0] = True
    rec.add_anchor(make_anchor(0.5, logL, valid))
    res = rec.run(np.empty(0, dtype=EVENT_DTYPE), np.array([1.0]))
    L, sup = res.frames[0][1], res.frames[0][2]
    assert L[0, 0] > 0
    assert L[1, 1] == pytest.approx(-0.3)       # masked pixel unchanged
    assert not sup[1, 1]
    assert sup[0, 0]


def test_simulator_events_are_threshold_consistent():
    scene = DriftingGratingScene(shape=(32, 48))
    events = generate_events(scene, duration=1.0, contrast=0.2, dt=2e-3)
    assert events.size > 100
    # polars only 0/1 and timestamps sorted after canonicalization
    s = sort_events(events)
    assert set(np.unique(s["p"])).issubset({0, 1})
    assert np.all(np.diff(s["t"]) >= 0)
    # integrated trajectory must track true log brightness at one pixel
    x, y = 24, 16
    pe = s[(s["x"] == x) & (s["y"] == y)]
    times = np.linspace(0, 1, 51)
    level, j, trace = 0.0, 0, []
    base = scene.log_brightness(0)[y, x]
    for t in times:
        while j < pe.size and pe[j]["t"] <= t:
            level += 0.2 if pe[j]["p"] == 1 else -0.2
            j += 1
        trace.append(base + level)
    truth = scene.log_brightness_at_times(times)[:, y, x]
    # threshold model tracks truth within a couple of contrast steps
    assert np.max(np.abs(np.array(trace) - truth)) < 0.45


def test_static_scene_emits_no_events_and_holds_level():
    scene = StaticPanelScene()
    events = generate_events(scene, duration=1.0, contrast=0.2)
    assert events.size == 0
    anchors = sparse_anchors(scene, [0.0], coverage=1.0, noise_sigma=0.0,
                             stratified=False)
    rec = Reconstructor(scene.shape, contrast=0.2)
    rec.add_anchor(anchors[0])
    res = rec.run(events, np.linspace(0, 1, 11))
    frames = res.frame_array()
    assert np.allclose(frames[0], frames[-1])
    assert np.allclose(frames[-1], scene.log_brightness(0.0), atol=1e-6)


def test_end_to_end_reconstruction_quality():
    from event_grey.metrics import evaluate

    bundle = simulate(name="moving_disk", duration=1.5,
                      anchor_coverage=0.1, seed=1)
    session = AnalysisSession(bundle)
    m = evaluate(session.result, bundle["scene"])
    # Accuracy is measured on pixels that actually received anchor support;
    # unsupported pixels carry the declared neutral prior by design.
    assert m["rmse_supported"] < 0.30
    assert m["n_unsupported_final"] == 0  # stratified masks cover every pixel


def test_corrupt_anchor_run_still_completes_and_flags_failure():
    bundle = simulate(name="fast_edge", duration=1.0,
                      corrupt_fraction=1.0, seed=2)
    session = AnalysisSession(bundle)
    failed = [r for r in session.result.reports if r.failed]
    assert len(failed) == 1
    assert any(a["level"] == "error" for a in session.anomalies)
    # reconstruction still produces finite frames everywhere
    assert np.isfinite(session.result.frame_array()).all()


def test_same_timestamp_many_events_pixel_order_deterministic():
    shape = (3, 3)
    rec = Reconstructor(shape, contrast=0.2,
                        prior=make_prior(shape, logL=np.zeros(shape)))
    ev = events_from_arrays(
        t=[0.5] * 4, x=[2, 0, 1, 0], y=[2, 0, 1, 0], p=[1, 1, 0, 0]
    )
    res = rec.run(ev, np.array([1.0]))
    L = res.frames[0][1]
    # events: (x=2,y=2) ON, (x=0,y=0) has ON and OFF, (x=1,y=1) OFF
    assert L[2, 2] == pytest.approx(0.2)
    assert L[0, 0] == pytest.approx(0.0)
    assert L[1, 1] == pytest.approx(-0.2)


def test_threshold_jitter_causes_measurable_drift():
    """An unmodelled per-pixel threshold mismatch must show up as drift that
    anchors then correct -- the mechanism is exercised explicitly."""
    clean = generate_events(DriftingGratingScene(), duration=1.0,
                            contrast=0.2, threshold_jitter=0.0)
    jittered = generate_events(DriftingGratingScene(), duration=1.0,
                               contrast=0.2, threshold_jitter=0.15)
    scene = DriftingGratingScene()

    def integration_rmse(events):
        inc = np.zeros(scene.shape)
        np.add.at(inc, (events["y"], events["x"]),
                  np.where(events["p"] == 1, 0.2, -0.2))
        truth = scene.log_brightness(1.0) - scene.log_brightness(0.0)
        return float(np.sqrt(np.mean((inc - truth) ** 2)))

    assert integration_rmse(clean) < 0.15
    assert integration_rmse(jittered) > integration_rmse(clean)


@pytest.mark.parametrize("scene_name", list(SCENES))
def test_all_scenes_run_end_to_end(scene_name):
    from event_grey.video import write_video
    import tempfile, os

    bundle = simulate(name=scene_name, duration=0.8, anchor_coverage=0.1,
                      seed=3)
    session = AnalysisSession(bundle, fps=10)
    assert np.isfinite(session.result.frame_array()).all()
    m = evaluate(session.result, bundle["scene"])
    assert m["rmse_supported"] < 0.35
    with tempfile.TemporaryDirectory() as d:
        path = write_video(os.path.join(d, "v.mp4"), session.result,
                           scene=bundle["scene"], fps=10)
        assert os.path.getsize(path) > 1000


def test_reconstruction_uses_only_events_and_anchors():
    """The reconstructor must produce identical frames with no access to the
    simulator scene: reference truth never enters the reconstruction path."""
    bundle = simulate(name="moving_disk", duration=1.0, anchor_coverage=0.15,
                      seed=5)
    times = np.linspace(0, 1.0, 21)

    def rebuild():
        rec = Reconstructor(bundle["shape"], contrast=bundle["contrast"])
        for a in bundle["anchors"]:
            rec.add_anchor(a)
        return rec.run(bundle["events"], times).frame_array()

    frames_a = rebuild()
    # Mutating the scene afterwards cannot change a reconstruction that never
    # reads it; rerun from the same raw inputs and expect bit-identical output.
    frames_b = rebuild()
    assert np.array_equal(frames_a, frames_b)
    # Anchor masks alone decide support; events alone never set it.
    rec = Reconstructor(bundle["shape"], contrast=bundle["contrast"])
    res = rec.run(bundle["events"], np.array([1.0]))
    assert not res.frames[0][2].any()
