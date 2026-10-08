from copy import deepcopy
from types import SimpleNamespace

import pytest

from photocull.project_store import ProjectStore
from photocull.scanner import ScannerService, ScanConflictError


def service(tmp_path):
    scanner = ScannerService(ProjectStore(tmp_path / 'projects'))
    scanner._results = {
        'schema_version': 2, 'rating_migration_status': 'native', 'project_id': 'project',
        'source_root': str(tmp_path / 'source'), 'identity_revision': 0,
        'photos': [
            {'id': 'a', 'stars': 3, 'rating_locked': True, 'person_ids': ['p1'],
             'faces': [{'face_id': 'a0', 'person_id': 'p1', 'bbox': [0, 0, .2, .2]}]},
            {'id': 'b', 'stars': 0, 'rating_locked': True, 'person_ids': ['p2'],
             'faces': [{'face_id': 'b0', 'person_id': 'p2', 'bbox': [.5, .5, .7, .7]}]},
        ], 'groups': [], 'summary': {'coverage_required_cells': 2},
        'coverage': {'required_cells': 2, 'unresolved_cells': 0},
        'rating_policy': {'required_coverage_keys': 2, 'unresolved_coverage_keys': 0},
        'engine': {'coverage_guard': {'required_coverage_keys': 2}},
    }
    return scanner


def merge(scanner, **kwargs):
    return scanner.correct_identities('project', 'merge', ['p1', 'p2'], [], 0, **kwargs)


def test_merge_persists_and_invalidates_coverage_without_changing_stars(tmp_path):
    scanner = service(tmp_path)
    result = merge(scanner)
    assert result['photos'][0]['person_ids'] == result['photos'][1]['person_ids']
    assert [p['stars'] for p in result['photos']] == [3, 0]
    assert all(p['rating_locked'] for p in result['photos'])
    assert result['identity_revision'] == 1
    assert result['identity_rescan_required'] is True
    assert result['coverage'] is None and result['rating_policy'] is None
    assert result['engine']['coverage_guard'] is None
    loaded = ScannerService(scanner.projects).load_project('project')
    assert loaded['identity_rescan_required'] is True
    assert loaded['photos'][0]['person_ids'] == result['photos'][0]['person_ids']


def test_merge_same_photo_refused_atomically(tmp_path):
    scanner = service(tmp_path)
    scanner._results['photos'][0]['faces'].append(deepcopy(scanner._results['photos'][1]['faces'][0]))
    original = deepcopy(scanner._results)
    with pytest.raises(ValueError, match='同一张'):
        merge(scanner)
    assert scanner._results == original


def test_stale_and_active_scan_commands_refused(tmp_path):
    scanner = service(tmp_path)
    with pytest.raises(ScanConflictError):
        scanner.correct_identities('old-project', 'merge', ['p1', 'p2'], [], 0)
    merge(scanner)
    with pytest.raises(ScanConflictError):
        merge(scanner)
    scanner._thread = SimpleNamespace(is_alive=lambda: True)
    with pytest.raises(ScanConflictError):
        scanner.correct_identities('project', 'split', [], [{'photo_id': 'a', 'face_id': 'a0'}], 1)


def test_split_and_replay_match_geometry_not_face_index_and_scope_source(tmp_path):
    scanner = service(tmp_path)
    scanner._results['photos'][1]['faces'][0]['person_id'] = 'p1'
    result = scanner.correct_identities('project', 'split', [], [{'photo_id': 'a', 'face_id': 'a0'}], 0)
    assert result['photos'][0]['person_ids'] != result['photos'][1]['person_ids']
    from photocull.identity_corrections import replay_corrections
    photos = [SimpleNamespace(id=p['id'], faces=[SimpleNamespace(face_id='new-index', bbox=f['bbox'], person_id='auto') for f in p['faces']], person_ids=[]) for p in result['photos']]
    assert replay_corrections(photos, result['identity_corrections']) == []
    assert photos[0].person_ids != photos[1].person_ids
    assert scanner.corrections.load(str(tmp_path / 'elsewhere')) == {}
    restarted = ScannerService(scanner.projects)
    assert restarted.corrections.load(result['source_root'])['identity_corrections'] == result['identity_corrections']


def test_manual_ratings_survive_source_rescan(tmp_path):
    scanner = service(tmp_path)
    scanner.rate_photo('a', 0)
    scanner.rate_photo('b', 3)
    state = ScannerService(scanner.projects).corrections.load(scanner._results['source_root'])
    assert state['manual_ratings']['a']['stars'] == 0
    assert state['manual_ratings']['b']['stars'] == 3


def test_old_saved_project_manual_locks_are_replayed(tmp_path):
    scanner = service(tmp_path)
    scanner.projects.save('project', scanner._results, {})
    state = scanner.corrections.load(scanner._results['source_root'])
    assert state['manual_ratings']['a'] == {'stars': 3, 'rating_locked': True}


def test_load_project_refused_during_scan(tmp_path):
    scanner = service(tmp_path)
    scanner.projects.save('project', scanner._results, {})
    scanner._thread = SimpleNamespace(is_alive=lambda: True)
    with pytest.raises(ScanConflictError):
        scanner.load_project('project')


def test_actual_legacy_project_recovers_source_from_files_and_relative_paths(tmp_path):
    scanner = service(tmp_path)
    source = scanner._results.pop('source_root')
    for photo in scanner._results['photos']:
        photo['relative_path'] = 'stage/' + photo['id'] + '.jpg'
    files = {p['id']: str(tmp_path / 'source' / p['relative_path']) for p in scanner._results['photos']}
    scanner.projects.save('project', scanner._results, files)
    assert scanner.corrections.load(source)['manual_ratings']['a']['stars'] == 3
    assert scanner.corrections.load(str(tmp_path / 'other')) == {}


def test_legacy_source_is_not_guessed_from_filename_only(tmp_path):
    scanner = service(tmp_path)
    source = scanner._results.pop('source_root')
    scanner.projects.save('project', scanner._results, {'a': source + '/a.jpg'})
    assert scanner.corrections.load(source) == {}


def test_recurring_correction_replaces_anchor_after_detector_geometry_shift(tmp_path):
    from photocull.identity_corrections import replay_corrections
    scanner = service(tmp_path)
    refs = [{'photo_id': 'a', 'face_id': 'a0'}]
    scanner.correct_identities('project', 'split', [], refs, 0)
    scanner._results['photos'][0]['faces'][0]['bbox'] = [0, 0, .21, .21]
    saved = scanner.correct_identities('project', 'split', [], refs, 1)
    face = SimpleNamespace(face_id='new', bbox=[0, 0, .21, .21], person_id='auto')
    photo = SimpleNamespace(id='a', faces=[face], person_ids=['auto'])
    anchors = [a for a in saved['identity_corrections'] if a['photo_id'] == 'a']
    assert replay_corrections([photo], anchors) == []
    assert photo.person_ids == saved['photos'][0]['person_ids']


def test_identity_summary_counts_final_repeated_identities(tmp_path):
    scanner = service(tmp_path)
    assert merge(scanner)['summary']['people'] == 1
    split = scanner.correct_identities('project', 'split', [], [{'photo_id': 'a', 'face_id': 'a0'}], 1)
    assert split['summary']['people'] == 0
