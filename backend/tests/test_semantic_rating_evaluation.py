from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def load_script() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts" / "evaluate-semantic-ratings.py"
    spec = importlib.util.spec_from_file_location("evaluate_semantic_ratings", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def payload() -> dict:
    return {
        "schema_version": 2,
        "rating_migration_status": "native",
        "lightroom_ready": True,
        "project_id": "semantic-test",
        "photos": [
            {
                "id": "primary",
                "filename": "A.jpg",
                "group_id": "group-a",
                "stars": 3,
                "rating_tier": "primary",
                "strict_duplicate_cluster_id": "cluster-a",
                "coverage_keys": [],
            },
            {
                "id": "coverage",
                "filename": "B.jpg",
                "group_id": "group-b",
                "stars": 2,
                "rating_tier": "coverage",
                "strict_duplicate_cluster_id": "cluster-b",
                "coverage_keys": ["stage-001:person-001"],
            },
            {
                "id": "valuable",
                "filename": "C.jpg",
                "group_id": "group-b",
                "stars": 1,
                "rating_tier": "valuable",
                "strict_duplicate_cluster_id": "cluster-c",
                "coverage_keys": [],
            },
            {
                "id": "waste",
                "filename": "D.jpg",
                "group_id": "group-d",
                "stars": 0,
                "rating_tier": "waste",
                "strict_duplicate_cluster_id": "cluster-d",
                "coverage_keys": [],
            },
        ],
        "rating_policy": {
            "required_coverage_keys": 1,
            "unresolved_coverage_keys": 0,
        },
    }


def test_three_star_metric_uses_only_primary() -> None:
    module = load_script()

    assert module.selected_ids(payload(), minimum_stars=3) == {"primary"}


def test_two_plus_three_metric_includes_coverage_not_valuable() -> None:
    module = load_script()

    assert module.selected_ids(payload(), minimum_stars=2) == {"primary", "coverage"}


def test_rating_metrics_use_exact_manual_members_and_report_coverage() -> None:
    module = load_script()

    metrics = module.evaluate_rating_set(
        payload(),
        manual_stems={"a", "b", "d"},
        minimum_stars=2,
    )

    assert metrics["photo_ids"] == ["coverage", "primary"]
    assert metrics["selected"] == 2
    assert metrics["manual_recall"] == 0.6667
    assert metrics["group_hit"] == 0.6667
    assert metrics["reduction"] == 0.5
    assert metrics["person_stage_coverage"] == 1.0


def test_strict_cluster_leaks_count_only_extra_selected_members() -> None:
    module = load_script()
    result = payload()
    result["photos"][1]["strict_duplicate_cluster_id"] = "cluster-a"
    result["photos"][1]["stars"] = 3

    metrics = module.evaluate_rating_set(result, manual_stems={"a", "b"}, minimum_stars=3)

    assert metrics["strict_cluster_leaks"] == 1
    assert metrics["strict_cluster_leak_clusters"] == ["cluster-a"]


def test_evaluator_rejects_legacy_projects() -> None:
    module = load_script()
    result = payload()
    result["rating_migration_status"] = "rescan_required"

    with pytest.raises(ValueError, match="重新扫描"):
        module.evaluate_rating_set(result, manual_stems={"a"}, minimum_stars=3)


def test_corrected_project_requires_rescan_before_semantic_evaluation():
    result = payload()
    result['rating_policy'] = None
    result['identity_rescan_required'] = True
    with pytest.raises(ValueError, match='重新扫描'):
        load_script().evaluate_rating_set(result, manual_stems=set(), minimum_stars=3)


def test_aggregate_markdown_reports_cross_event_freeze_guards() -> None:
    module = load_script()
    aggregate = {
        "events": ["活动A-iqa", "活动B-iqa"],
        "same_frozen_model": True,
        "no_runtime_fit": True,
        "no_per_event_overrides": True,
        "semantic_contract_valid": True,
        "rating_sets": {
            "three_star": {
                "selected": 10,
                "group_hit": 0.8,
                "manual_recall": 0.6,
                "reduction": 0.35,
                "strict_cluster_leaks": 0,
                "person_stage_coverage": 1.0,
            },
            "two_plus_three_star": {
                "selected": 12,
                "group_hit": 0.9,
                "manual_recall": 0.7,
                "reduction": 0.25,
                "strict_cluster_leaks": 1,
                "person_stage_coverage": 1.0,
            },
        },
    }

    markdown = module.render_aggregate_markdown(aggregate)

    assert "活动A-iqa、活动B-iqa" in markdown
    assert "冻结模型一致：通过" in markdown
    assert "运行时拟合：无" in markdown


def test_coverage_metric_is_recomputed_for_final_star_scope() -> None:
    module = load_script()
    result = payload()
    result['rating_policy']['unresolved_coverage_keys'] = 0  # Scan metadata says covered by 2-star.
    metrics = module.evaluate_rating_set(result, manual_stems=set(), minimum_stars=3)
    assert metrics['person_stage_coverage'] == 0.0
    assert metrics['unresolved_person_stage_keys'] == 1


def test_report_distinguishes_evaluator_and_saved_scan_provenance(tmp_path) -> None:
    import json
    module = load_script()
    project_file = tmp_path / 'project.json'
    project_file.write_text(json.dumps(payload()))
    reference = tmp_path / 'reference'
    reference.mkdir()
    (reference / 'A.jpg').write_bytes(b'')
    model_file = Path(__file__).resolve().parents[1] / 'photocull/assets/rating_model_v1.json'
    report = module.build_report(event_name='test', project_file=project_file,
                                 reference_dir=reference, rating_model_file=model_file)
    assert len(report['provenance']['evaluation_commit']) == 40
    assert report['provenance']['scan_provenance'] is None
    assert report['independent_pair_audit'] is None
    assert report['cluster_metric_scope'] == 'stored_cluster_consistency_only'


def test_aggregate_cannot_infer_saved_scan_model_from_current_evaluator(tmp_path) -> None:
    import json
    module = load_script()
    project_file = tmp_path / 'project.json'
    project_file.write_text(json.dumps(payload()))
    reference = tmp_path / 'reference'
    reference.mkdir()
    model_file = Path(__file__).resolve().parents[1] / 'photocull/assets/rating_model_v1.json'
    report = module.build_report(event_name='test', project_file=project_file,
                                 reference_dir=reference, rating_model_file=model_file)
    aggregate = module.aggregate_reports([report])
    assert aggregate['same_evaluator_model'] is True
    assert aggregate['same_frozen_model'] is None
    assert aggregate['no_runtime_fit'] is None
    assert aggregate['no_per_event_overrides'] is None
    markdown = module.render_aggregate_markdown(aggregate)
    assert '冻结模型一致：未知' in markdown
    assert '运行时拟合：未知' in markdown
