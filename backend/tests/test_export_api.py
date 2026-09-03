from fastapi.testclient import TestClient

import photocull.project_store as project_store_module
from photocull.exporter import ExportOperationStore
from photocull.project_store import ProjectStore
from photocull.scanner import ScannerService


def test_two_phase_export_requires_confirmation_and_is_idempotent(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(project_store_module, "PROJECTS_DIR", tmp_path / "api-runtime" / "projects")
    import photocull.api as api

    source = tmp_path / "source" / "photo.jpg"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"photo-bytes")
    destination = tmp_path / "destination"
    scanner = ScannerService(ProjectStore(tmp_path / "projects"))
    scanner._results = {
        "schema_version": 2,
        "rating_migration_status": "native",
        "lightroom_ready": True,
        "project_id": "api-export",
        "photos": [{"id": "photo", "relative_path": "photo.jpg", "stars": 3}],
        "groups": [],
        "summary": {},
    }
    scanner._files = {"photo": str(source)}
    monkeypatch.setattr(api, "scanner", scanner)
    monkeypatch.setattr(
        api,
        "export_operations",
        ExportOperationStore(tmp_path / "operations"),
    )
    monkeypatch.delenv("PHOTOCULL_API_TOKEN", raising=False)
    client = TestClient(api.app)

    preflight = client.post(
        "/api/exports/preflights",
        json={"destination": str(destination), "project_id": "api-export", "minimum_stars": 2},
    )

    assert preflight.status_code == 200
    payload = preflight.json()
    assert payload["copy_count"] == 1
    assert not destination.exists()
    operation_id = payload["operation_id"]
    assert client.post(
        f"/api/exports/{operation_id}/execute",
        json={"plan_hash": payload["plan_hash"], "confirmed": False},
    ).status_code == 422

    executed = client.post(
        f"/api/exports/{operation_id}/execute",
        json={"plan_hash": payload["plan_hash"], "confirmed": True},
    )
    repeated = client.post(
        f"/api/exports/{operation_id}/execute",
        json={"plan_hash": payload["plan_hash"], "confirmed": True},
    )

    assert executed.status_code == 200
    assert executed.json()["copied"] == 1
    assert executed.json()["verification_passed"] is True
    assert repeated.json() == executed.json()
    assert (destination / "3星精选" / "photo.jpg").read_bytes() == b"photo-bytes"
