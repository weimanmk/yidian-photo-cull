from fastapi.testclient import TestClient

import photocull.project_store as project_store_module
from photocull.project_store import ProjectStore
from photocull.scanner import ScannerService


def test_rating_endpoint_persists_manual_semantic_star(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(project_store_module, "PROJECTS_DIR", tmp_path / "api-runtime" / "projects")
    import photocull.api as api

    scanner = ScannerService(ProjectStore(tmp_path / "projects"))
    scanner._results = {
        "schema_version": 2,
        "rating_migration_status": "native",
        "lightroom_ready": True,
        "project_id": "api-rating",
        "photos": [
            {
                "id": "photo-a",
                "group_id": "group-1",
                "category": "duplicate",
                "is_best_pick": False,
                "stars": 0,
                "rating_tier": "waste",
                "rating_origin": "ai",
                "rating_reason": "redundant_reject",
                "rating_locked": False,
                "coverage_keys": [],
            }
        ],
        "groups": [{"id": "group-1", "photo_ids": ["photo-a"], "best_photo_ids": []}],
        "rating_policy": {"required_coverage_keys": 0, "unresolved_coverage_keys": 0},
        "summary": {},
    }
    monkeypatch.setattr(api, "scanner", scanner)
    monkeypatch.delenv("PHOTOCULL_API_TOKEN", raising=False)
    client = TestClient(api.app)

    response = client.patch("/api/photos/photo-a/rating", json={"stars": 3, "locked": True})

    assert response.status_code == 200
    assert scanner.results()["photos"][0]["rating_tier"] == "primary"
    assert scanner.results()["photos"][0]["rating_locked"] is True
    assert client.patch("/api/photos/photo-a/rating", json={"stars": 4}).status_code == 422
