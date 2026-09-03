from __future__ import annotations

import json

import pytest

from photocull.lightroom_service import LightroomPlanError, LightroomService, LightroomStateError
from tests.lightroom_fixtures import service_with_project, write_execute_receipt, write_preflight_receipt


def write_completed_execution_journal(
    service: LightroomService,
    operation,
    *,
    incomplete_item_id: str | None = None,
) -> None:
    assert operation.execute_manifest is not None
    assert operation.preflight_receipt is not None
    receipt = operation.preflight_receipt
    progress = {}
    for item in receipt.items:
        if item.action not in {"new", "update"}:
            continue
        entry = {"verified": item.item_id != incomplete_item_id}
        if item.action == "new":
            entry["catalog_added"] = True
        progress[item.item_id] = entry
    payload = {
        "schema_version": 1,
        "operation_id": operation.id,
        "preflight_request_id": operation.execute_manifest.preflight_request_id,
        "plan_hash": operation.plan_hash,
        "baseline_hash": receipt.baseline_hash,
        "catalog_name": receipt.catalog_name,
        "catalog_identity_hash": receipt.catalog_identity_hash,
        "counts": receipt.counts.model_dump(mode="json"),
        "items": [item.model_dump(mode="json") for item in receipt.items],
        "progress": progress,
        "revision": 7,
    }
    (service.bridge.journals / f"{operation.id}.00000007.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_preflight_refuses_legacy_project(tmp_path) -> None:
    service = service_with_project(tmp_path, lightroom_ready=False)

    with pytest.raises(LightroomPlanError, match="重新扫描"):
        service.create_preflight("legacy-project")


def test_execute_requires_matching_verified_preflight(tmp_path) -> None:
    service = service_with_project(tmp_path, lightroom_ready=True)
    operation = service.create_preflight("v2-project")

    assert operation.status == "waiting_for_plugin"
    assert len(operation.preflight_manifest.items) == 3
    assert {item.target_rating for item in operation.preflight_manifest.items} == {0, 2, 3}
    with pytest.raises(LightroomStateError):
        service.confirm_execute(operation.id)

    write_preflight_receipt(service.bridge, operation, counts={"new": 2, "update": 1})
    refreshed = service.refresh(operation.id)

    assert refreshed.status == "awaiting_confirmation"
    assert refreshed.can_execute is True
    executing = service.confirm_execute(operation.id)
    assert executing.status == "executing"
    assert executing.execute_request_id is not None
    assert (service.bridge.inbox / f"{executing.execute_request_id}.json").is_file()


def test_invalid_preflight_items_disable_confirmation(tmp_path) -> None:
    service = service_with_project(tmp_path, lightroom_ready=True)
    operation = service.create_preflight("v2-project")
    write_preflight_receipt(service.bridge, operation, counts={"invalid": 1})

    refreshed = service.refresh(operation.id)

    assert refreshed.status == "awaiting_confirmation"
    assert refreshed.can_execute is False
    with pytest.raises(LightroomStateError, match="无效"):
        service.confirm_execute(operation.id)


def test_preflight_rejects_missing_authoritative_file_mapping(tmp_path) -> None:
    service = service_with_project(tmp_path, lightroom_ready=True, missing_file_mapping=True)

    with pytest.raises(LightroomPlanError, match="源文件映射"):
        service.create_preflight("v2-project")

    assert list(service.bridge.inbox.glob("*.json")) == []


def test_confirmation_rejects_receipt_changed_after_refresh(tmp_path) -> None:
    service = service_with_project(tmp_path, lightroom_ready=True)
    operation = service.create_preflight("v2-project")
    receipt_path = write_preflight_receipt(service.bridge, operation)
    refreshed = service.refresh(operation.id)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["catalog_identity_hash"] = "d" * 64
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LightroomStateError, match="目录|基线"):
        service.confirm_execute(refreshed.id)


def test_operation_survives_service_restart(tmp_path) -> None:
    service = service_with_project(tmp_path, lightroom_ready=True)
    operation = service.create_preflight("v2-project")
    restarted = LightroomService(service.projects, service.bridge, app_version="0.2.1")

    loaded = restarted.get(operation.id)

    assert loaded.id == operation.id
    assert loaded.plan_hash == operation.plan_hash
    persisted = json.loads((service.bridge.operations / f"{operation.id}.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "waiting_for_plugin"


def test_confirmation_rejects_source_fingerprint_drift(tmp_path) -> None:
    service = service_with_project(tmp_path, lightroom_ready=True)
    operation = service.create_preflight("v2-project")
    write_preflight_receipt(service.bridge, operation)
    service.refresh(operation.id)
    source = operation.preflight_manifest.items[0].source_path
    with open(source, "ab") as stream:
        stream.write(b"changed")

    with pytest.raises(LightroomStateError, match="源文件"):
        service.confirm_execute(operation.id)


def test_complete_receipt_is_read_back_before_operation_completes(tmp_path) -> None:
    service = service_with_project(tmp_path, lightroom_ready=True)
    operation = service.create_preflight("v2-project")
    write_preflight_receipt(service.bridge, operation, counts={"new": 2, "update": 1})
    service.refresh(operation.id)
    executing = service.confirm_execute(operation.id)
    write_execute_receipt(service.bridge, executing)

    completed = service.refresh(operation.id)

    assert completed.status == "complete"
    assert completed.execute_receipt is not None
    assert completed.execute_receipt.counts.verified == 3
    assert completed.execute_receipt.counts.pending_rating == 0
    assert service.request_rollback(operation.id).status == "rollback_preflight"


def test_running_operation_exposes_latest_journal_progress(tmp_path) -> None:
    service = service_with_project(tmp_path, lightroom_ready=True)
    operation = service.create_preflight("v2-project")
    write_preflight_receipt(service.bridge, operation, counts={"new": 2, "update": 1})
    service.refresh(operation.id)
    executing = service.confirm_execute(operation.id)
    actionable = [item for item in executing.preflight_receipt.items if item.action in {"new", "update"}]
    write_completed_execution_journal(service, executing, incomplete_item_id=actionable[0].item_id)

    running = service.refresh(operation.id)
    counts = running.public_dict()["counts"]

    assert running.status == "executing"
    assert running.execute_receipt is None
    assert counts["catalog_added"] == 2
    assert counts["verified"] == 2
    assert counts["pending_rating"] == 0


def test_pending_rating_receipt_never_reports_complete(tmp_path) -> None:
    service = service_with_project(tmp_path, lightroom_ready=True)
    operation = service.create_preflight("v2-project")
    write_preflight_receipt(service.bridge, operation, counts={"new": 2, "update": 1})
    service.refresh(operation.id)
    executing = service.confirm_execute(operation.id)
    write_execute_receipt(service.bridge, executing, status="pending_rating", pending_rating=1)

    pending = service.refresh(operation.id)

    assert pending.status == "pending_rating"
    assert pending.execute_receipt is not None
    assert pending.execute_receipt.counts.verified == 2


def test_membership_drift_receipt_recovers_from_completed_journal(tmp_path) -> None:
    service = service_with_project(tmp_path, lightroom_ready=True)
    operation = service.create_preflight("v2-project")
    write_preflight_receipt(service.bridge, operation, counts={"new": 2, "update": 1})
    preflight = service.refresh(operation.id)
    executing = service.confirm_execute(preflight.id)
    receipt_path = write_execute_receipt(service.bridge, executing, status="replan_required")
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["error_code"] = "catalog_membership_drift"
    payload["error_message"] = "Lightroom 目录基线已变化"
    payload["counts"]["catalog_added"] = 1
    payload["counts"]["verified"] = 2
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    write_completed_execution_journal(service, executing)

    completed = service.refresh(operation.id)

    assert completed.status == "complete"
    assert completed.error_code is None
    assert completed.execute_receipt is not None
    assert completed.execute_receipt.status == "complete"
    assert completed.execute_receipt.counts.catalog_added == 2
    assert completed.execute_receipt.counts.verified == 3
    assert completed.execute_receipt.counts.pending_rating == 0


def test_membership_drift_does_not_recover_from_partial_journal(tmp_path) -> None:
    service = service_with_project(tmp_path, lightroom_ready=True)
    operation = service.create_preflight("v2-project")
    write_preflight_receipt(service.bridge, operation, counts={"new": 2, "update": 1})
    preflight = service.refresh(operation.id)
    executing = service.confirm_execute(preflight.id)
    receipt_path = write_execute_receipt(service.bridge, executing, status="replan_required")
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["error_code"] = "catalog_membership_drift"
    payload["error_message"] = "Lightroom 目录基线已变化"
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    incomplete_item_id = next(item.item_id for item in preflight.preflight_receipt.items if item.action == "new")
    write_completed_execution_journal(service, executing, incomplete_item_id=incomplete_item_id)

    failed = service.refresh(operation.id)

    assert failed.status == "failed"
    assert failed.error_code == "catalog_membership_drift"


def test_pending_rating_retry_reuses_verified_plan_with_a_new_request_id(tmp_path) -> None:
    service = service_with_project(tmp_path, lightroom_ready=True)
    operation = service.create_preflight("v2-project")
    write_preflight_receipt(service.bridge, operation, counts={"new": 2, "update": 1})
    service.refresh(operation.id)
    executing = service.confirm_execute(operation.id)
    write_execute_receipt(service.bridge, executing, status="pending_rating", pending_rating=1)
    pending = service.refresh(operation.id)

    retrying = service.retry_pending_rating(operation.id)

    assert retrying.status == "executing"
    assert retrying.execute_request_id is not None
    assert retrying.execute_request_id != executing.execute_request_id
    assert retrying.execute_manifest is not None
    assert executing.execute_manifest is not None
    assert retrying.execute_manifest.operation_id == operation.id
    assert retrying.execute_manifest.plan_hash == operation.plan_hash
    assert retrying.execute_manifest.preflight_request_id == executing.execute_manifest.preflight_request_id
    assert retrying.execute_manifest.baseline_hash == executing.execute_manifest.baseline_hash
    assert retrying.execute_manifest.catalog_identity_hash == executing.execute_manifest.catalog_identity_hash
    assert retrying.execute_manifest.items == executing.execute_manifest.items
    assert retrying.execute_receipt == pending.execute_receipt
    assert (service.bridge.inbox / f"{retrying.execute_request_id}.json").is_file()


def test_pending_rating_retry_accepts_chunk_evidence_when_legacy_count_is_zero(tmp_path) -> None:
    service = service_with_project(tmp_path, lightroom_ready=True)
    operation = service.create_preflight("v2-project")
    write_preflight_receipt(service.bridge, operation, counts={"new": 2, "update": 1})
    service.refresh(operation.id)
    executing = service.confirm_execute(operation.id)
    receipt_path = write_execute_receipt(
        service.bridge,
        executing,
        status="pending_rating",
        pending_rating=0,
    )
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["chunks"] = [
        {
            "chunk_id": "new-0001",
            "status": "pending_rating",
            "counts": {"total": 1},
            "error_code": "pending_rating",
        }
    ]
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")

    pending = service.refresh(operation.id)
    retrying = service.retry_pending_rating(operation.id)

    assert pending.status == "pending_rating"
    assert retrying.status == "executing"
    assert retrying.execute_request_id != executing.execute_request_id


def test_retry_rejects_operations_without_pending_rating(tmp_path) -> None:
    service = service_with_project(tmp_path, lightroom_ready=True)
    operation = service.create_preflight("v2-project")

    with pytest.raises(LightroomStateError, match="等待星级"):
        service.retry_pending_rating(operation.id)
