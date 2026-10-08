from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load_audit():
    spec = importlib.util.spec_from_file_location('duplicate_audit', ROOT / 'scripts/audit-duplicate-leakage.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def project():
    return {'schema_version': 2, 'rating_migration_status': 'native', 'lightroom_ready': True,
            'project_id': 'offline-test', 'photos': [
                {'id': 'a', 'stars': 3, 'category': 'rejected', 'group_id': 'one', 'strict_duplicate_cluster_id': 'x'},
                {'id': 'b', 'stars': 3, 'category': 'rejected', 'group_id': 'two', 'strict_duplicate_cluster_id': 'y'},
                {'id': 'c', 'stars': 2, 'category': 'selected', 'group_id': 'two', 'strict_duplicate_cluster_id': 'y'},
                {'id': 'd', 'stars': 0, 'category': 'selected', 'group_id': 'one', 'strict_duplicate_cluster_id': 'x'},
            ], 'groups': [{'id': 'one'}, {'id': 'two'}]}


def labels():
    # a/b missed by production; b/c falsely merged by production.
    return {'schema_version': 1, 'label_source': 'human', 'labeler': 'reviewer', 'pairs': [
        {'left_id': 'a', 'right_id': 'b', 'duplicate': True},
        {'left_id': 'b', 'right_id': 'c', 'duplicate': False},
        {'left_id': 'a', 'right_id': 'd', 'duplicate': True},
    ]}


def test_audit_uses_final_stars_not_legacy_category():
    report = load_audit().audit_project(project(), minimum_stars=3)
    assert report['photo_ids'] == ['a', 'b']
    assert report['strict_cluster_leak_pairs'] == 0
    assert report['independent_pair_audit'] is None


def test_independent_labels_reveal_missed_leaks_and_false_merges():
    report = load_audit().audit_project(project(), minimum_stars=2, pair_labels=labels())
    metrics = report['independent_pair_audit']
    assert metrics['false_negative_pairs'] == [['a', 'b']]
    assert metrics['false_positive_pairs'] == [['b', 'c']]
    assert metrics['selected_duplicate_pairs'] == [['a', 'b']]
    assert metrics['precision'] == 0.5
    assert metrics['recall'] == 0.5
    assert metrics['scope'] == 'labeled_pairs_only'


@pytest.mark.parametrize('bad_pair', [
    {'left_id': 'a', 'right_id': 'missing', 'duplicate': True},
    {'left_id': 'a', 'right_id': 'a', 'duplicate': True},
    {'left_id': 'a', 'right_id': 'b', 'duplicate': 'false'},
])
def test_bad_pair_labels_are_rejected(bad_pair):
    ground_truth = labels()
    ground_truth['pairs'] = [bad_pair]
    with pytest.raises(ValueError):
        load_audit().audit_project(project(), minimum_stars=3, pair_labels=ground_truth)


def test_offline_cli_has_actual_file_hashes_and_runtime_provenance(tmp_path):
    project_file = tmp_path / 'project.json'
    pair_file = tmp_path / 'pairs.json'
    project_file.write_text(json.dumps({'results': project()}))
    pair_file.write_text(json.dumps(labels()))
    completed = subprocess.run([sys.executable, str(ROOT / 'scripts/audit-duplicate-leakage.py'),
                                '--project-file', str(project_file), '--pair-labels', str(pair_file)],
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    import hashlib
    assert report['project_sha256'] == hashlib.sha256(project_file.read_bytes()).hexdigest().upper()
    assert report['pair_labels_sha256'] == hashlib.sha256(pair_file.read_bytes()).hexdigest().upper()
    assert len(report['provenance']['evaluation_commit']) == 40
    assert len(report['provenance']['rating_model_sha256']) == 64
    assert report['provenance']['duplicate_config']['version']
    assert report['provenance']['scan_provenance'] is None  # Do not invent the saved project's origin.


def test_repeated_or_contradictory_unordered_labels_are_rejected():
    ground_truth = labels()
    ground_truth['pairs'].append({'left_id': 'b', 'right_id': 'a', 'duplicate': False})
    with pytest.raises(ValueError, match='pair label'):
        load_audit().audit_project(project(), pair_labels=ground_truth)


def test_production_labels_cannot_claim_independent_human_ground_truth():
    ground_truth = labels()
    ground_truth['label_source'] = 'production_comparator'
    with pytest.raises(ValueError, match='human'):
        load_audit().audit_project(project(), pair_labels=ground_truth)


def test_corrected_project_requires_rescan_before_duplicate_audit():
    result = project()
    result['identity_rescan_required'] = True
    with pytest.raises(ValueError, match='重新扫描'):
        load_audit().audit_project(result)


def test_scan_provenance_binds_running_pipeline_model_and_settings():
    from photocull.audit import scan_provenance
    from photocull.scanner import FEATURE_PIPELINE_VERSION
    value = scan_provenance(settings={'face_identity_threshold': .42},
                            feature_pipeline_version=FEATURE_PIPELINE_VERSION,
                            pipeline_signature='signature')
    assert value['feature_pipeline_version'] == FEATURE_PIPELINE_VERSION
    assert value['pipeline_signature'] == 'signature'
    assert len(value['rating_model_sha256']) == 64
    assert value['settings']['face_identity_threshold'] == .42
    assert value['duplicate_config']['version']
    assert value['runtime_fit'] is False
    assert value['per_event_overrides'] == {}
