"""Quantitative evaluation against simulator reference frames.

Reference frames are used here strictly for offline scoring.  They are never
passed to the reconstructor; the metrics measure how well the event
integration plus sparse anchors recovers the true continuous field.
"""

from __future__ import annotations

import numpy as np


def evaluate(result, scene, reference_times=None):
    """Compute drift, static-region and fast-edge error metrics."""
    if reference_times is None:
        reference_times = result.times()
    ref_times = np.asarray(reference_times, dtype=np.float64)
    ref_log = scene.log_brightness_at_times(ref_times)

    out_times = result.times()
    rec = result.frame_array()
    # Align reference samples to the reconstruction grid.
    idx = np.searchsorted(ref_times, out_times)
    idx = np.clip(idx, 0, len(ref_times) - 1)
    ref_aligned = ref_log[idx]

    err = rec - ref_aligned
    supported = result.supported_masks()

    metrics = {}
    metrics["rmse_log"] = float(np.sqrt(np.mean(err ** 2)))
    metrics["mae_log"] = float(np.mean(np.abs(err)))
    if supported.any():
        metrics["rmse_supported"] = float(np.sqrt(np.mean(err[supported] ** 2)))
        metrics["mae_supported"] = float(np.mean(np.abs(err[supported])))
    else:
        metrics["rmse_supported"] = float("nan")
        metrics["mae_supported"] = float("nan")
    # Headline accuracy: only pixels that have actually received an anchor
    # or supported prior.  Unsupported pixels carry the explicit neutral
    # prior and are scored separately, never presented as measured output.
    metrics["mean_support_fraction"] = float(supported.mean())

    # Brightness drift: how the mean error evolves over the run.  Anchors
    # should pull it back toward zero; unbounded growth means lost tracking.
    metrics["drift_curve"] = np.mean(err, axis=(1, 2))
    metrics["max_abs_drift"] = float(np.max(np.abs(metrics["drift_curve"])))

    # Static-region check: pixels whose true brightness barely changes.
    ref_var = np.var(ref_aligned, axis=0)
    static_mask = ref_var < 1e-8
    n_static = int(static_mask.sum())
    metrics["n_static_pixels"] = n_static
    if static_mask.any():
        # Reconstruction should be constant there too.
        metrics["static_pixel_variance"] = float(
            np.mean(np.var(rec, axis=0)[static_mask])
        )
    else:
        metrics["static_pixel_variance"] = 0.0

    # Fast-edge check: error restricted to supported pixels where the true
    # spatial gradient magnitude is large.
    gx = np.abs(np.diff(ref_aligned, axis=2, prepend=ref_aligned[:, :, :1]))
    edge_mask = (gx > np.quantile(gx, 0.95)) & supported
    if edge_mask.any():
        metrics["edge_mae"] = float(np.mean(np.abs(err[edge_mask])))
    else:
        metrics["edge_mae"] = 0.0

    # Static-region absolute error likewise excludes unsupported prior pixels.
    static_supported = static_mask & supported[-1]
    if static_supported.any():
        metrics["static_final_mae"] = float(
            np.mean(np.abs(err[-1][static_supported]))
        )
    else:
        metrics["static_final_mae"] = 0.0

    metrics["n_unsupported_final"] = int((~supported[-1]).sum())
    metrics["n_anchors"] = len(result.anchors)
    metrics["n_failed_anchors"] = sum(1 for r in result.reports if r.failed)
    metrics["n_resets"] = sum(r.reset_pixels for r in result.reports)
    return metrics


def format_metrics(metrics):
    lines = [
        f"RMSE (log brightness, supported):    {metrics['rmse_supported']:.4f}",
        f"MAE  (log brightness, supported):    {metrics['mae_supported']:.4f}",
        f"RMSE including unsupported priors:   {metrics['rmse_log']:.4f}",
        f"mean anchor/prior support fraction:  {metrics['mean_support_fraction']:.2%}",
        f"max |mean drift|:                    {metrics['max_abs_drift']:.4f}",
        f"static-region variance:              {metrics['static_pixel_variance']:.2e}",
        f"static-region final MAE:             {metrics['static_final_mae']:.4f}",
        f"fast-edge MAE (supported):           {metrics['edge_mae']:.4f}",
        f"unsupported final pixels:            {metrics['n_unsupported_final']}",
        f"anchors / failed / resets:           {metrics['n_anchors']} / "
        f"{metrics['n_failed_anchors']} / {metrics['n_resets']}",
    ]
    return "\n".join(lines)
