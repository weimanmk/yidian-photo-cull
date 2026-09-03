from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def load_script() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts" / "evaluate-human-reference.py"
    spec = importlib.util.spec_from_file_location("evaluate_human_reference", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def photo(
    filename: str,
    group_id: str,
    rank: int,
    *,
    issues: list[str] | None = None,
    metrics: dict[str, float] | None = None,
) -> dict[str, object]:
    return {
        "id": filename,
        "filename": filename,
        "group_id": group_id,
        "rank_in_group": rank,
        "is_best_pick": rank == 1,
        "issues": list(issues or []),
        "metrics": dict(metrics or {}),
    }


def test_eye_diagnostics_counts_manual_hard_kills_and_eye_related_disagreements() -> None:
    module = load_script()
    photos = [
        photo("AI_0001.RW2", "group-1", 1, metrics={"eye_score": 94.0}),
        photo(
            "MANUAL_0002.RW2",
            "group-1",
            2,
            issues=["主要人物闭眼"],
            metrics={
                "eye_score": 54.0,
                "decisive_closed_face_count": 1.0,
                "uncertain_eye_face_count": 0.0,
            },
        ),
        photo(
            "MANUAL_0003.RW2",
            "group-2",
            1,
            metrics={
                "eye_score": 72.0,
                "decisive_closed_face_count": 0.0,
                "uncertain_eye_face_count": 1.0,
            },
        ),
    ]

    diagnostics = module.eye_evidence_diagnostics(
        photos,
        {"manual_0002", "manual_0003"},
    )

    assert diagnostics == {
        "manual_with_decisive_closed": 1,
        "manual_with_uncertain_eye": 1,
        "eye_related_top1_disagreements": 1,
        "top1_disagreements": 1,
        "eye_related_disagreement_rate": 1.0,
    }


def test_eye_diagnostics_uses_v020_issue_only_when_new_metric_is_missing() -> None:
    module = load_script()
    legacy_manual = photo(
        "LEGACY_0001.RW2",
        "group-1",
        1,
        issues=["主要人物闭眼"],
        metrics={"eye_score": 20.0},
    )
    candidate_manual = photo(
        "CANDIDATE_0002.RW2",
        "group-2",
        1,
        issues=["主要人物闭眼"],
        metrics={"eye_score": 20.0, "decisive_closed_face_count": 0.0},
    )

    diagnostics = module.eye_evidence_diagnostics(
        [legacy_manual, candidate_manual],
        {"legacy_0001", "candidate_0002"},
    )

    assert diagnostics["manual_with_decisive_closed"] == 1


def test_unknown_eye_state_does_not_create_eye_diagnostic_counts() -> None:
    module = load_script()
    manual = photo("MANUAL_0001.RW2", "group-1", 1, metrics={"eye_score": 72.0})

    diagnostics = module.eye_evidence_diagnostics([manual], {"manual_0001"})

    assert diagnostics["manual_with_decisive_closed"] == 0
    assert diagnostics["manual_with_uncertain_eye"] == 0
    assert diagnostics["eye_related_top1_disagreements"] == 0
    assert diagnostics["top1_disagreements"] == 0
    assert diagnostics["eye_related_disagreement_rate"] == 0.0


def test_semantic_rating_metrics_separate_primary_and_delivery_sets() -> None:
    module = load_script()
    photos = [
        {**photo("PRIMARY.RW2", "group-1", 1), "stars": 3, "rating_tier": "primary"},
        {**photo("COVERAGE.RW2", "group-2", 1), "stars": 2, "rating_tier": "coverage"},
        {**photo("VALUABLE.RW2", "group-3", 1), "stars": 1, "rating_tier": "valuable"},
        {**photo("WASTE.RW2", "group-4", 1), "stars": 0, "rating_tier": "waste"},
    ]

    metrics = module.semantic_rating_metrics(
        photos,
        {"primary", "coverage"},
        {"primary", "coverage", "valuable", "waste"},
    )

    assert metrics["available"] is True
    assert metrics["star_counts"] == {"0": 1, "1": 1, "2": 1, "3": 1}
    assert metrics["three_star"]["predicted"] == 1
    assert metrics["three_star"]["recall"] == 0.5
    assert metrics["two_plus_three_star"]["predicted"] == 2
    assert metrics["two_plus_three_star"]["recall"] == 1.0
