from fastapi.testclient import TestClient
from test_identity_corrections import service


def test_identity_endpoint_persists_and_rejects_stale_commands(tmp_path, monkeypatch):
    import photocull.api as api
    scanner = service(tmp_path)
    monkeypatch.setattr(api, 'scanner', scanner)
    monkeypatch.delenv('PHOTOCULL_API_TOKEN', raising=False)
    client = TestClient(api.app)
    payload = {'operation': 'merge', 'person_ids': ['p1', 'p2'], 'faces': [], 'revision': 0}
    response = client.post('/api/projects/project/identities/corrections', json=payload)
    assert response.status_code == 200
    assert response.json()['identity_rescan_required'] is True
    assert response.json()['photos'][0]['stars'] == 3
    assert client.post('/api/projects/project/identities/corrections', json=payload).status_code == 409
    assert client.post('/api/projects/other/identities/corrections', json=payload).status_code == 409
    payload['revision'] = 1
    payload['operation'] = 'invalid'
    assert client.post('/api/projects/project/identities/corrections', json=payload).status_code == 422


def test_identity_endpoint_same_photo_conflict_does_not_mutate(tmp_path, monkeypatch):
    import photocull.api as api
    scanner = service(tmp_path)
    scanner._results['photos'][0]['faces'].append({'face_id': 'a1', 'person_id': 'p2', 'bbox': [.5, .5, .7, .7]})
    monkeypatch.setattr(api, 'scanner', scanner)
    monkeypatch.delenv('PHOTOCULL_API_TOKEN', raising=False)
    response = TestClient(api.app).post('/api/projects/project/identities/corrections', json={
        'operation': 'merge', 'person_ids': ['p1', 'p2'], 'revision': 0,
    })
    assert response.status_code == 400
    assert '同一张' in response.json()['detail']
    assert scanner.results()['identity_revision'] == 0


def test_rating_cannot_race_a_rescan_snapshot(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import photocull.api as api
    scanner = service(tmp_path)
    scanner._thread = SimpleNamespace(is_alive=lambda: True)
    monkeypatch.setattr(api, 'scanner', scanner)
    monkeypatch.delenv('PHOTOCULL_API_TOKEN', raising=False)
    response = TestClient(api.app).patch('/api/photos/a/rating', json={'stars': 0, 'locked': True})
    assert response.status_code == 409
    assert scanner.results()['photos'][0]['stars'] == 3
