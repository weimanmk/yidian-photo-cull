"""Offline evaluation of saved decisions against independently supplied labels."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import subprocess
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / 'backend/photocull/assets/rating_model_v1.json'


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def scan_provenance(*, settings: dict, feature_pipeline_version: str,
                    pipeline_signature: str) -> dict:
    """Record scan-time evidence; unavailable packaged sources stay unknown."""
    from .near_duplicates import DUPLICATE_CONFIG
    from .rating_model import DEFAULT_MODEL_PATH
    from .rating_policy import MIN_PRIMARY_SCORE, PRIMARY_BLOCKING_ISSUES
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                        stderr=subprocess.DEVNULL, text=True, timeout=5).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT,
                                            stderr=subprocess.DEVNULL, text=True, timeout=5).strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    source_dir = Path(__file__).resolve().parent
    sources = ['scanner.py', 'rating_policy.py', 'near_duplicates.py', 'identity.py',
               'identity_corrections.py', 'face_engine.py', 'face_quality.py', 'coverage.py']
    return {
        'commit': commit, 'worktree_dirty': dirty,
        'feature_pipeline_version': feature_pipeline_version,
        'pipeline_signature': pipeline_signature,
        'rating_model_sha256': file_sha256(DEFAULT_MODEL_PATH) if DEFAULT_MODEL_PATH.is_file() else None,
        'duplicate_config': dict(DUPLICATE_CONFIG),
        'quality_eligibility': {'minimum_score': MIN_PRIMARY_SCORE,
                                'blocking_issues': sorted(PRIMARY_BLOCKING_ISSUES)},
        'settings': dict(settings),
        'source_sha256': {name: file_sha256(source_dir / name) if (source_dir / name).is_file() else None
                         for name in sources},
        'runtime_fit': False, 'per_event_overrides': {},
    }


def evaluation_provenance(result: dict, model_file: Path = DEFAULT_MODEL) -> dict:
    # Reading the literal records the current evaluator configuration without
    # loading production image/model dependencies or treating it as ground truth.
    source = ROOT / 'backend/photocull/near_duplicates.py'
    config = next(ast.literal_eval(node.value) for node in ast.parse(source.read_text()).body
                  if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'DUPLICATE_CONFIG' for t in node.targets))
    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(['git', *args], cwd=ROOT, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    commit = git('rev-parse', 'HEAD')
    status = git('status', '--porcelain')
    return {
        'evaluation_commit': commit,
        'evaluation_worktree_dirty': status != '' if status is not None else None,
        'rating_model_sha256': file_sha256(model_file),
        'duplicate_config': config,
        'duplicate_source_sha256': file_sha256(source),
        'evaluator_source_sha256': {
            str(path.relative_to(ROOT)): file_sha256(path)
            for path in (Path(__file__).resolve(), ROOT / 'scripts/audit-duplicate-leakage.py',
                         ROOT / 'scripts/evaluate-semantic-ratings.py')
        },
        'scan_provenance': result.get('scan_provenance'),
        'scope': 'current_evaluator_files_not_saved_scan_origin',
    }


def audit_project(result: dict, *, minimum_stars: int = 3,
                  pair_labels: dict | None = None) -> dict[str, Any]:
    if result.get('identity_rescan_required'):
        raise ValueError('人物身份已修正，请重新扫描后再评测')
    if (result.get('schema_version') != 2 or result.get('rating_migration_status') != 'native'
            or result.get('lightroom_ready') is not True):
        raise ValueError('项目不是原生语义星级，请重新扫描后再评测')
    if minimum_stars not in (2, 3):
        raise ValueError('minimum_stars must be 2 or 3')
    photos = {str(p['id']): p for p in result['photos']}
    if len(photos) != len(result['photos']):
        raise ValueError('Project contains duplicate photo IDs')
    selected = {key for key, p in photos.items() if int(p.get('stars', 0)) >= minimum_stars}
    clusters: dict[str, list[str]] = {}
    for key in sorted(selected):
        cluster = photos[key].get('strict_duplicate_cluster_id')
        if cluster:
            clusters.setdefault(str(cluster), []).append(key)
    leaks = {key: members for key, members in clusters.items() if len(members) > 1}
    report = {
        'project_id': result.get('project_id'), 'minimum_stars': minimum_stars,
        'photo_scope': f'stars_at_least_{minimum_stars}', 'photo_ids': sorted(selected),
        'photo_count': len(selected), 'cluster_metric_scope': 'stored_cluster_consistency_only',
        'strict_cluster_leaks': sum(len(m) - 1 for m in leaks.values()),
        'strict_cluster_leak_pairs': sum(len(m) * (len(m) - 1) // 2 for m in leaks.values()),
        'strict_cluster_leak_clusters': sorted(leaks), 'independent_pair_audit': None,
    }
    if pair_labels is None:
        return report
    if (not isinstance(pair_labels, dict) or pair_labels.get('schema_version') != 1
            or pair_labels.get('label_source') != 'human'
            or not isinstance(pair_labels.get('labeler'), str) or not pair_labels['labeler'].strip()
            or not isinstance(pair_labels.get('pairs'), list)):
        raise ValueError('Expected schema_version=1, label_source=human, labeler and pairs')
    counts = {'true_positive': 0, 'false_positive': 0, 'false_negative': 0, 'true_negative': 0}
    false_positive, false_negative, selected_duplicates = [], [], []
    seen = set()
    for pair in pair_labels['pairs']:
        if not isinstance(pair, dict):
            raise ValueError('Each pair must be an object')
        left, right, duplicate = pair.get('left_id'), pair.get('right_id'), pair.get('duplicate')
        if (not isinstance(left, str) or not isinstance(right, str)
                or left not in photos or right not in photos or left == right or type(duplicate) is not bool):
            raise ValueError('Pairs require distinct existing photo IDs and a boolean duplicate label')
        key = tuple(sorted((left, right)))
        if key in seen:
            raise ValueError(f'Duplicate or contradictory pair label: {key}')
        seen.add(key)
        left_cluster = photos[left].get('strict_duplicate_cluster_id')
        predicted = bool(left_cluster and left_cluster == photos[right].get('strict_duplicate_cluster_id'))
        outcome = ('true_positive' if predicted else 'false_negative') if duplicate else (
            'false_positive' if predicted else 'true_negative')
        counts[outcome] += 1
        if outcome == 'false_positive':
            false_positive.append(list(key))
        if outcome == 'false_negative':
            false_negative.append(list(key))
        if duplicate and left in selected and right in selected:
            selected_duplicates.append(list(key))
    tp, fp, fn = (counts[key] for key in ('true_positive', 'false_positive', 'false_negative'))
    report['independent_pair_audit'] = {
        'scope': 'labeled_pairs_only', 'label_source': 'human', 'labeler': pair_labels['labeler'],
        'labeled_pairs': len(seen), **{f'{key}_count': value for key, value in counts.items()},
        'false_positive_pairs': sorted(false_positive), 'false_negative_pairs': sorted(false_negative),
        'selected_duplicate_pairs': sorted(selected_duplicates),
        'selected_duplicate_pair_count': len(selected_duplicates),
        'precision': round(tp / (tp + fp), 4) if tp + fp else None,
        'recall': round(tp / (tp + fn), 4) if tp + fn else None,
    }
    return report
