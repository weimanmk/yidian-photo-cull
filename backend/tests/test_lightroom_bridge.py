from __future__ import annotations

import json

from photocull.config import lightroom_bridge_root
from photocull.lightroom_bridge import LightroomBridge
from tests.lightroom_fixtures import manifest


def test_submit_never_exposes_partial_json(tmp_path) -> None:
    bridge = LightroomBridge(tmp_path / "bridge")
    request = manifest(tmp_path)

    target = bridge.submit(request)

    assert target.parent.name == "inbox"
    assert not list(target.parent.glob("*.tmp"))
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["operation"] == "preflight"
    assert payload["plan_hash"] == request.plan_hash


def test_read_receipt_validates_outbox_payload(tmp_path) -> None:
    bridge = LightroomBridge(tmp_path / "bridge")
    request = manifest(tmp_path)
    receipt_payload = {
        "schema_version": 1,
        "request_id": request.request_id,
        "operation_id": request.operation_id,
        "plan_hash": request.plan_hash,
        "baseline_hash": "b" * 64,
        "catalog_name": "Yidian Smoke",
        "catalog_identity_hash": "c" * 64,
        "started_at": "2026-08-30T10:00:00+08:00",
        "finished_at": "2026-08-30T10:00:01+08:00",
        "status": "awaiting_confirmation",
        "counts": {
            "total": 1,
            "new": 1,
            "update": 0,
            "unchanged": 0,
            "protected": 0,
            "invalid": 0,
            "catalog_added": 0,
            "pending_rating": 0,
            "verified": 0,
            "rolled_back": 0,
        },
        "chunks": [],
        "items": [
            {
                "item_id": "photo-1",
                "path_hash": request.items[0].path_hash,
                "action": "new",
                "previous_rating": None,
                "target_rating": 3,
                "final_rating": None,
                "status": "planned",
            }
        ],
    }
    outbox_path = bridge.outbox / f"{request.request_id}.json"
    outbox_path.write_text(json.dumps(receipt_payload), encoding="utf-8")

    receipt = bridge.read_receipt(request.request_id)

    assert receipt is not None
    assert receipt.status == "awaiting_confirmation"
    assert receipt.counts.new == 1


def test_read_latest_journal_selects_highest_numeric_revision(tmp_path) -> None:
    bridge = LightroomBridge(tmp_path / "bridge")
    request = manifest(tmp_path)
    for revision in (2, 17):
        payload = {"operation_id": request.operation_id, "revision": revision}
        (bridge.journals / f"{request.operation_id}.{revision:08d}.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

    journal = bridge.read_latest_journal(request.operation_id)

    assert journal == {"operation_id": request.operation_id, "revision": 17}


def test_recover_processing_returns_claimed_request_to_inbox(tmp_path) -> None:
    bridge = LightroomBridge(tmp_path / "bridge")
    request = manifest(tmp_path)
    processing_path = bridge.processing / f"{request.request_id}.json"
    processing_path.write_bytes(request.model_dump_json().encode("utf-8"))

    recovered = bridge.recover_processing()

    assert recovered == [bridge.inbox / processing_path.name]
    assert recovered[0].is_file()
    assert not processing_path.exists()


def test_lightroom_bridge_root_honors_test_override(monkeypatch, tmp_path) -> None:
    configured = tmp_path / "bridge-root"
    monkeypatch.setenv("PHOTOCULL_LIGHTROOM_BRIDGE_DIR", str(configured))

    assert lightroom_bridge_root() == configured.resolve()


def test_lightroom_bridge_root_matches_lightroom_sdk_app_data(monkeypatch, tmp_path) -> None:
    local_app_data = tmp_path / "Local"
    roaming_app_data = tmp_path / "Roaming"
    monkeypatch.delenv("PHOTOCULL_LIGHTROOM_BRIDGE_DIR", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
    monkeypatch.setenv("APPDATA", str(roaming_app_data))
    monkeypatch.setattr("photocull.config.platform.system", lambda: "Windows")

    assert lightroom_bridge_root() == (
        roaming_app_data / "Adobe" / "Lightroom" / "YidianPhotoCull" / "lightroom-bridge"
    ).resolve()
