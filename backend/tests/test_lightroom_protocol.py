from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from photocull.lightroom_protocol import (
    LightroomExecuteManifest,
    LightroomPreflightManifest,
    LightroomReceipt,
    canonical_json,
    plan_hash,
)
from tests.lightroom_fixtures import fixture, manifest


def test_plan_hash_ignores_created_at_and_request_identity_but_changes_for_target_rating(tmp_path) -> None:
    first = manifest(
        tmp_path,
        created_at="2026-08-30T10:00:00+08:00",
        rating=3,
        request_id="1" * 32,
        operation_id="2" * 32,
    )
    second = manifest(
        tmp_path,
        created_at="2026-08-30T11:00:00+08:00",
        rating=3,
        request_id="3" * 32,
        operation_id="4" * 32,
    )
    changed = manifest(
        tmp_path,
        created_at="2026-08-30T11:00:00+08:00",
        rating=2,
        request_id="3" * 32,
        operation_id="4" * 32,
    )

    assert plan_hash(first) == plan_hash(second)
    assert plan_hash(first) != plan_hash(changed)


def test_manifest_rejects_rating_outside_zero_to_three(tmp_path) -> None:
    with pytest.raises(ValidationError):
        manifest(tmp_path, rating=4)


def test_manifest_rejects_non_sha256_path_hash(tmp_path) -> None:
    valid = manifest(tmp_path)
    payload = valid.model_dump(mode="json")
    payload["items"][0]["path_hash"] = "not-a-hash"

    with pytest.raises(ValidationError):
        type(valid).model_validate(payload)


def test_plan_hash_is_independent_of_item_input_order(tmp_path) -> None:
    original = manifest(tmp_path)
    second_item = original.items[0].model_copy(
        update={
            "item_id": "photo-2",
            "source_path": original.items[0].source_path.replace("source.jpg", "other.jpg"),
            "path_hash": "a" * 64,
            "target_rating": 2,
            "rating_tier": "coverage",
        }
    )
    forward = original.model_copy(update={"items": [original.items[0], second_item]})
    reverse = original.model_copy(update={"items": [second_item, original.items[0]]})

    assert plan_hash(forward) == plan_hash(reverse)


def test_canonical_json_is_utf8_stable_and_compact() -> None:
    encoded = canonical_json({"z": 1, "name": "一点筛图", "a": False})

    assert encoded == b'{"a":false,"name":"\xe4\xb8\x80\xe7\x82\xb9\xe7\xad\x9b\xe5\x9b\xbe","z":1}'
    assert json.loads(encoded.decode("utf-8"))["name"] == "一点筛图"


def test_lua_preflight_request_fixture_matches_python_plan_hash() -> None:
    request = LightroomPreflightManifest.model_validate_json(fixture("preflight-request.json"))

    assert request.operation == "preflight"
    assert request.plan_hash == plan_hash(request)
    assert len(request.items) == 5


def test_lua_preflight_fixture_validates_against_python_receipt_model() -> None:
    receipt = LightroomReceipt.model_validate_json(fixture("preflight-receipt.json"))

    assert receipt.status == "awaiting_confirmation"
    assert receipt.counts.model_dump(mode="json") == {
        "total": 5,
        "new": 1,
        "update": 1,
        "unchanged": 1,
        "protected": 1,
        "invalid": 1,
        "catalog_added": 0,
        "pending_rating": 0,
        "verified": 0,
        "rolled_back": 0,
    }
    assert {item.action for item in receipt.items} == {"new", "update", "unchanged", "protected", "invalid"}


def test_lua_execute_fixture_keeps_the_verified_preflight_plan() -> None:
    request = LightroomExecuteManifest.model_validate_json(fixture("execute-request.json"))

    assert request.operation == "execute"
    assert request.plan_hash == plan_hash(request)
    assert request.preflight_request_id == "1" * 32
    assert request.baseline_hash == "b" * 64
    assert request.catalog_identity_hash == "c" * 64


def test_complete_fixture_requires_every_mutation_to_be_verified() -> None:
    receipt = LightroomReceipt.model_validate_json(fixture("complete-receipt.json"))

    assert receipt.status == "complete"
    assert receipt.counts.verified == receipt.counts.new + receipt.counts.update
    assert receipt.counts.pending_rating == 0
    assert receipt.counts.catalog_added == receipt.counts.new
