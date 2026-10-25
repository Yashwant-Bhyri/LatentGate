import pytest

from research.evaluate_m0_feedback_mechanism import evaluate


def test_saved_m0_x0_preview_produces_a_bounded_counterfactual_mask_update():
    report = evaluate()
    assert report["status"] == "offline_counterfactual_diagnostic_no_model_rerun"
    assert report["final_output_vs_commanded_path"]["scoreable_coverage"] == 1.0
    assert report["final_output_vs_commanded_path"]["fraction_of_commanded_horizontal_displacement"] == pytest.approx(
        0.0978363, abs=1e-6
    )
    step4 = report["step4_x0_hat_vs_commanded_path"]
    assert step4["observer_scoreable_coverage"] == 1.0
    assert step4["trusted_unlocked_action_box_count_of_4"] == 4
    assert step4["rasterized_changed_token_total"] > 0
    assert "do not re-run denoising" in report["method"]["counterfactual"]
