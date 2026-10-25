import math

import pytest

from research.evaluate_ditraj_red_observer_kubric import center_error_metrics, evaluate


def test_center_error_scales_normalized_axes_before_euclidean_distance():
    # A 1% normalized shift on both axes is not isotropic in pixels on a 160x96 frame.
    px, diagonal, components = center_error_metrics([0.51, 0.51], [0.5, 0.5], 160, 96)
    expected_px = math.hypot(1.6, 0.96)
    assert components == pytest.approx([1.6, 0.96])
    assert px == pytest.approx(expected_px)
    assert diagonal == pytest.approx(expected_px / math.hypot(160, 96))
    assert diagonal == pytest.approx(0.01)


def test_kubric_calibration_reports_raw_start_and_controller_frames_separately():
    report = evaluate()
    aggregate = report["aggregate"]["all"]
    assert aggregate["scoreable_coverage"] == 1.0
    assert aggregate["mean_valid_center_error_px"] == pytest.approx(1.729739, abs=1e-5)
    assert aggregate["first_frame_absolute_center_error_mean_px_missing_penalized"] == pytest.approx(
        1.761329, abs=1e-5
    )
    assert aggregate["controller_keyframe_count"] == 10
    assert aggregate["controller_keyframe_anchor_relative_mean_l2_control_grid_units"] == pytest.approx(
        0.0243352, abs=1e-6
    )
    assert "not a quantized token error" in report["fixture"]["relative_error_units"]
