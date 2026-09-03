from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def load_script() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts" / "compare-ranking-evaluations.py"
    spec = importlib.util.spec_from_file_location("compare_ranking_evaluations", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summary(
    *,
    top1: float = 0.70,
    top2: float = 0.90,
    strict_f1: float = 0.62,
    coverage: float = 0.95,
    dedup: float = 0.55,
    decisive_closed: int = 2,
) -> dict[str, object]:
    required_cells = 10_000
    unresolved_cells = round((1.0 - coverage) * required_cells)
    return {
        "group_evaluation": {
            "top1_exact_group_hit_rate": top1,
            "group_hit_at_2": top2,
        },
        "strict_ai_best": {"f1": strict_f1},
        "grouping_summary": {"top1_reduction_rate": dedup},
        "engine": {
            "coverage_guard": {
                "required_cells": required_cells,
                "unresolved_cells": unresolved_cells,
            }
        },
        "eye_evidence_evaluation": {
            "manual_with_decisive_closed": decisive_closed,
        },
    }


def comparison(module: ModuleType, name: str, baseline: dict[str, object], candidate: dict[str, object]):
    return module.compare_dataset(name, baseline, candidate)


def test_development_gate_accepts_three_activity_mean_top1_gain_of_three_points() -> None:
    module = load_script()
    comparisons = [
        comparison(module, name, summary(), summary(top1=0.73))
        for name in ("活动A", "活动B", "活动C")
    ]

    report = module.evaluate_gates("development", comparisons)

    assert report["passed"] is True
    assert report["mean_top1_delta"] == pytest.approx(0.03)


def test_development_gate_rejects_one_activity_top1_drop_beyond_one_point() -> None:
    module = load_script()
    comparisons = [
        comparison(module, "活动A", summary(), summary(top1=0.689)),
        comparison(module, "活动B", summary(), summary(top1=0.80)),
        comparison(module, "活动C", summary(), summary(top1=0.80)),
    ]

    report = module.evaluate_gates("development", comparisons)

    assert report["passed"] is False
    assert any(check["name"] == "活动A.top1_not_down_more_than_1pp" and not check["passed"] for check in report["checks"])


@pytest.mark.parametrize("metric", ["top2", "strict_f1", "coverage", "dedup"])
def test_development_gate_rejects_safety_metric_drop_beyond_one_point(metric: str) -> None:
    module = load_script()
    candidate_values = {
        "top1": 0.74,
        "top2": 0.90,
        "strict_f1": 0.62,
        "coverage": 0.95,
        "dedup": 0.55,
    }
    candidate_values[metric] -= 0.011
    failed = summary(**candidate_values)
    comparisons = [
        comparison(module, "活动A", summary(), failed),
        comparison(module, "活动B", summary(), summary(top1=0.74)),
        comparison(module, "活动C", summary(), summary(top1=0.74)),
    ]

    report = module.evaluate_gates("development", comparisons)

    assert report["passed"] is False
    assert any(check["name"] == f"活动A.{metric}_not_down_more_than_1pp" and not check["passed"] for check in report["checks"])


def test_development_gate_rejects_increased_decisive_eye_kills() -> None:
    module = load_script()
    comparisons = [
        comparison(module, "活动A", summary(decisive_closed=2), summary(top1=0.74, decisive_closed=3)),
        comparison(module, "活动B", summary(), summary(top1=0.74)),
        comparison(module, "活动C", summary(), summary(top1=0.74)),
    ]

    report = module.evaluate_gates("development", comparisons)

    assert report["passed"] is False
    assert any(check["name"] == "活动A.decisive_eye_kills_not_increased" and not check["passed"] for check in report["checks"])


def test_holdout_requires_strict_top1_improvement_and_non_decreasing_top2() -> None:
    module = load_script()
    equal_top1 = [comparison(module, "holdout", summary(top1=0.78), summary(top1=0.78))]
    lower_top2 = [comparison(module, "holdout", summary(top1=0.78), summary(top1=0.79, top2=0.899))]

    assert module.evaluate_gates("holdout", equal_top1)["passed"] is False
    assert module.evaluate_gates("holdout", lower_top2)["passed"] is False


def test_holdout_reports_eighty_percent_target_without_using_it_as_a_gate() -> None:
    module = load_script()
    below_target = [comparison(module, "holdout", summary(top1=0.78), summary(top1=0.79))]
    at_target = [comparison(module, "holdout", summary(top1=0.78), summary(top1=0.80))]

    below_report = module.evaluate_gates("holdout", below_target)
    target_report = module.evaluate_gates("holdout", at_target)

    assert below_report["passed"] is True
    assert below_report["target_80_met"] is False
    assert target_report["passed"] is True
    assert target_report["target_80_met"] is True


def test_mode_requires_exact_number_of_dataset_pairs() -> None:
    module = load_script()
    one = [comparison(module, "one", summary(), summary(top1=0.74))]

    with pytest.raises(ValueError, match="至少三个"):
        module.evaluate_gates("development", one)
    with pytest.raises(ValueError, match="恰好一个"):
        module.evaluate_gates("holdout", one + one)
