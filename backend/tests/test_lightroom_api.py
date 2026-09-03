from __future__ import annotations

from fastapi.testclient import TestClient

import photocull.project_store as project_store_module
from tests.lightroom_fixtures import service_with_project, write_execute_receipt, write_preflight_receipt


def test_lightroom_api_exposes_two_phase_operation_without_accepting_manifest(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(project_store_module, "PROJECTS_DIR", tmp_path / "api-runtime" / "projects")
    import photocull.api as api

    service = service_with_project(tmp_path, lightroom_ready=True)
    monkeypatch.setattr(api, "lightroom_service", service)
    monkeypatch.delenv("PHOTOCULL_API_TOKEN", raising=False)
    client = TestClient(api.app)

    status = client.get("/api/lightroom/status")
    created = client.post("/api/lightroom/preflights", json={"project_id": "v2-project"})

    assert status.status_code == 200
    assert status.json()["bridge_root"] == str(service.bridge.root)
    assert created.status_code == 200
    payload = created.json()
    assert payload["status"] == "waiting_for_plugin"
    assert payload["item_count"] == 3
    assert "preflight_manifest" not in payload
    operation_id = payload["id"]
    assert client.get(f"/api/lightroom/operations/{operation_id}").status_code == 200
    assert client.post(f"/api/lightroom/operations/{operation_id}/execute").status_code == 409

    operation = service.get(operation_id)
    write_preflight_receipt(service.bridge, operation, counts={"new": 2, "update": 1})
    refreshed = client.get(f"/api/lightroom/operations/{operation_id}")
    executed = client.post(f"/api/lightroom/operations/{operation_id}/execute")

    assert refreshed.status_code == 200
    assert refreshed.json()["can_execute"] is True
    assert refreshed.json()["counts"]["new"] == 2
    assert executed.status_code == 200
    assert executed.json()["status"] == "executing"


def test_lightroom_api_returns_explicit_project_operation_and_state_errors(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(project_store_module, "PROJECTS_DIR", tmp_path / "api-runtime" / "projects")
    import photocull.api as api

    service = service_with_project(tmp_path, lightroom_ready=False)
    monkeypatch.setattr(api, "lightroom_service", service)
    monkeypatch.delenv("PHOTOCULL_API_TOKEN", raising=False)
    client = TestClient(api.app)

    assert client.post("/api/lightroom/preflights", json={"project_id": "legacy-project"}).status_code == 400
    assert client.get(f"/api/lightroom/operations/{'f' * 32}").status_code == 404
    assert client.post(f"/api/lightroom/operations/{'f' * 32}/execute").status_code == 404
    assert client.post(f"/api/lightroom/operations/{'f' * 32}/retry").status_code == 404


def test_lightroom_api_retries_only_pending_rating_operations(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(project_store_module, "PROJECTS_DIR", tmp_path / "api-runtime" / "projects")
    import photocull.api as api

    service = service_with_project(tmp_path, lightroom_ready=True)
    monkeypatch.setattr(api, "lightroom_service", service)
    monkeypatch.delenv("PHOTOCULL_API_TOKEN", raising=False)
    client = TestClient(api.app)

    created = client.post("/api/lightroom/preflights", json={"project_id": "v2-project"})
    operation_id = created.json()["id"]
    operation = service.get(operation_id)
    write_preflight_receipt(service.bridge, operation, counts={"new": 2, "update": 1})
    assert client.get(f"/api/lightroom/operations/{operation_id}").status_code == 200
    executing = client.post(f"/api/lightroom/operations/{operation_id}/execute")
    write_execute_receipt(service.bridge, service.get(operation_id), status="pending_rating", pending_rating=1)
    assert client.get(f"/api/lightroom/operations/{operation_id}").json()["status"] == "pending_rating"

    retried = client.post(f"/api/lightroom/operations/{operation_id}/retry")

    assert executing.status_code == 200
    assert retried.status_code == 200
    assert retried.json()["status"] == "executing"
    assert retried.json()["execute_request_id"] != executing.json()["execute_request_id"]
    assert (service.bridge.inbox / f"{retried.json()['execute_request_id']}.json").is_file()
    assert client.post(f"/api/lightroom/operations/{operation_id}/retry").status_code == 409
