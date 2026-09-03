from __future__ import annotations

import json

from tests.lightroom_fixtures import PLUGIN_ROOT, all_plugin_source, plugin_file


def test_plugin_declares_supported_sdk_and_background_entry() -> None:
    info = plugin_file("Info.lua")

    assert "LrSdkVersion = 15.0" in info
    assert "LrSdkMinimumVersion = 14.3" in info
    assert 'LrToolkitIdentifier = "com.yidian.photocull.lightroom"' in info
    assert 'LrPluginName = "一点筛图"' in info
    assert 'LrInitPlugin = "InitPlugin.lua"' in info
    assert "LrForceInitPlugin = true" in info
    assert 'LrShutdownPlugin = "ShutdownPlugin.lua"' in info
    assert 'file = "ManualCheck.lua"' in info


def test_plugin_version_manifest_matches_release() -> None:
    manifest = json.loads((PLUGIN_ROOT / "manifest.json").read_text(encoding="utf-8"))
    info = plugin_file("Info.lua")

    assert manifest["schema_version"] == 1
    assert manifest["plugin_id"] == "com.yidian.photocull.lightroom"
    assert manifest["version"] == "0.2.1"
    assert "VERSION = {" in info
    assert "major = 0" in info
    assert "minor = 2" in info
    assert "revision = 1" in info


def test_plugin_owns_single_background_watcher_and_shutdown_state() -> None:
    init = plugin_file("InitPlugin.lua")
    shutdown = plugin_file("ShutdownPlugin.lua")
    state = plugin_file("PluginState.lua")

    assert init.count("LrTasks.startAsyncTask") == 1
    assert "PluginState.started" in init
    assert "LrTasks.sleep(2)" in init
    assert "PluginState.running = false" in shutdown
    assert 'plugin-heartbeat.json' in state
    assert "pcall" in init


def test_plugin_reload_replaces_stale_background_watcher_generation() -> None:
    init = plugin_file("InitPlugin.lua")
    shutdown = plugin_file("ShutdownPlugin.lua")
    state = plugin_file("PluginState.lua")

    assert "generation = 0" in state
    assert "PluginState.generation = PluginState.generation + 1" in init
    assert "local generation = PluginState.generation" in init
    assert "PluginState.generation == generation" in init
    assert "if not PluginState.started then" not in init
    assert "PluginState.generation = PluginState.generation + 1" in shutdown


def test_plugin_uses_lightroom_epoch_conversion_and_sdk_file_cleanup() -> None:
    state = plugin_file("PluginState.lua")
    receipt = plugin_file("Receipt.lua")
    source = all_plugin_source()

    timestamp_expression = "LrDate.timeToPosixDate(LrDate.currentTime())"
    assert timestamp_expression in state
    assert timestamp_expression in receipt
    assert "os.remove" not in source
    assert "LrFileUtils.delete(temporary)" in state
    assert "LrFileUtils.delete(temporary)" in plugin_file("Bridge.lua")


def test_repeating_heartbeat_replaces_only_its_existing_outbox_file() -> None:
    state = plugin_file("PluginState.lua")

    assert 'PluginState.writeAtomicOutbox("plugin-heartbeat.json"' in state
    assert "LrFileUtils.delete(target)" in state


def test_heartbeat_publication_serializes_same_outbox_target() -> None:
    state = plugin_file("PluginState.lua")
    publication = state[state.index("function PluginState.writeAtomicOutbox") : state.index("function PluginState.writeHeartbeat")]

    assert "local outboxWriteLeases = {}" in state
    assert "local function acquireOutboxWrite(target)" in state
    assert "while outboxWriteLeases[target] do" in state
    assert "LrTasks.sleep(0.01)" in state
    assert publication.index("acquireOutboxWrite(target)") < publication.index("LrFileUtils.delete(target)")
    assert publication.index("releaseOutboxWrite(target)") > publication.index("LrFileUtils.delete(target)")


def test_background_watcher_surfaces_bridge_load_failures() -> None:
    init = plugin_file("InitPlugin.lua")

    assert "if not loaded then" in init
    assert "error(Bridge)" in init


def test_plugin_error_boundaries_allow_lightroom_sdk_calls_to_yield() -> None:
    init = plugin_file("InitPlugin.lua")
    manual = plugin_file("ManualCheck.lua")
    bridge = plugin_file("Bridge.lua")
    operation = plugin_file("CatalogOperation.lua")

    assert "LrTasks.pcall" in init
    assert "LrTasks.pcall" in manual
    assert 'local LrTasks = import "LrTasks"' in bridge
    assert "local ok, result = LrTasks.pcall(function()" in bridge
    assert 'local LrTasks = import "LrTasks"' in operation
    assert "local writeOk = LrTasks.pcall(function()" in operation
    assert "local writeOk, writeError = LrTasks.pcall(function()" in operation
    assert "local ok, bridgeError = pcall(Bridge.processOnce)" not in manual


def test_atomic_file_publication_allows_lightroom_sdk_calls_to_yield() -> None:
    state = plugin_file("PluginState.lua")
    bridge = plugin_file("Bridge.lua")

    assert 'local LrTasks = import "LrTasks"' in state
    assert "LrTasks.pcall(function()" in state
    assert "local ok, writeError = LrTasks.pcall(function()" in bridge
    assert "local ok, writeError = pcall(function()" not in state
    assert "local ok, writeError = pcall(function()" not in bridge


def test_plugin_contains_no_network_collection_or_shell_capability() -> None:
    source = all_plugin_source()

    for forbidden in (
        "LrHttp",
        "createCollection",
        "createCollectionSet",
        "executeCommand",
        "powershell",
        "cmd.exe",
        "moveToTrash",
    ):
        assert forbidden not in source


def test_plugin_bundles_json_codec_supported_by_lightroom_runtime() -> None:
    manifest = json.loads((PLUGIN_ROOT / "manifest.json").read_text(encoding="utf-8"))
    source = all_plugin_source()

    assert 'import "LrJson"' not in source
    assert "Json.lua" in manifest["files"]
    assert 'local Json = require "Json"' in plugin_file("PluginState.lua")
    assert 'local Json = require "Json"' in plugin_file("Bridge.lua")


def test_bridge_claims_requests_and_publishes_receipts_with_same_directory_temporary_files() -> None:
    bridge = plugin_file("Bridge.lua")

    assert 'LrPathUtils.getStandardFilePath("appData")' in bridge
    assert "function Bridge.claimNext" in bridge
    assert "function Bridge.complete" in bridge
    assert '"processing"' in bridge
    assert '"quarantine"' in bridge
    assert '".tmp"' in bridge
    assert "LrFileUtils.move" in bridge


def test_bridge_resumes_claimed_processing_request_after_plugin_reload() -> None:
    bridge = plugin_file("Bridge.lua")
    claim_next = bridge[bridge.index("function Bridge.claimNext") : bridge.index("local function serializable")]

    assert 'candidatesIn("processing")' in claim_next
    assert claim_next.index('candidatesIn("processing")') < claim_next.index('candidatesIn("inbox")')


def test_bridge_serializes_background_and_manual_request_consumers() -> None:
    bridge = plugin_file("Bridge.lua")
    claim_next = bridge[bridge.index("function Bridge.claimNext") : bridge.index("local function serializable")]
    process_once = bridge[bridge.index("function Bridge.processOnce") :]

    assert "local inFlightRequests = {}" in bridge
    assert "if not requestInFlight(candidate) then" in claim_next
    assert "claimRequest(candidate)" in claim_next
    assert "local function quarantineClaimed" in claim_next
    assert "local moved = LrFileUtils.move(candidate, target)" in claim_next
    assert process_once.count("releaseRequest(requestPath)") == 2


def test_preflight_reads_catalog_and_file_fingerprints_without_write_access() -> None:
    operation = plugin_file("CatalogOperation.lua")
    preflight = operation[
        operation.index("function CatalogOperation.preflight") : operation.index("local function indexById")
    ]

    assert "function CatalogOperation.preflight" in preflight
    assert "LrFileUtils.fileAttributes" in operation
    assert "catalog:findPhotoByPath" in operation
    assert 'getRawMetadata("rating")' in operation
    assert "file_size" in operation
    assert "modified_ns" in operation
    assert "withWriteAccessDo" not in preflight


def test_receipt_never_copies_absolute_source_path() -> None:
    receipt = plugin_file("Receipt.lua")

    assert "item_id" in receipt
    assert "path_hash" in receipt
    assert "source_path" not in receipt


def test_plugin_revalidates_catalog_baseline_before_any_write() -> None:
    operation = plugin_file("CatalogOperation.lua")

    assert "function CatalogOperation.revalidateBaseline" in operation
    assert "replan_required" in operation
    assert operation.index("CatalogOperation.revalidateBaseline") < operation.index("withWriteAccessDo")


def test_plugin_uses_batched_updates_and_serialized_import_chunks_with_immediate_journal() -> None:
    operation = plugin_file("CatalogOperation.lua")
    write_chunk = operation[operation.index("local function writeChunk") : operation.index("local function refreshReceipt")]

    assert "UPDATE_CHUNK_SIZE = 100" in operation
    assert "NEW_CHUNK_SIZE = 1" in operation
    assert "catalog:addPhoto(item.source_path)" in operation
    assert "catalog_added = true" in operation
    assert write_chunk.index("catalog_added = true") < write_chunk.index('setRawMetadata("rating"')
    assert "catalog:findPhotoByPath(item.source_path)" in operation


def test_plugin_never_yields_to_journal_io_inside_catalog_write_access() -> None:
    operation = plugin_file("CatalogOperation.lua")
    write_chunk = operation[operation.index("local function writeChunk") : operation.index("local function refreshReceipt")]
    rollback = operation[operation.index("function CatalogOperation.rollback") :]

    write_access_start = write_chunk.index('catalog:withWriteAccessDo("一点筛图 · 写入星级"')
    write_access_body = write_chunk[
        write_access_start : write_chunk.index("end)", write_access_start)
    ]
    rollback_write_access_start = rollback.index('catalog:withWriteAccessDo("一点筛图 · 撤销本次星级"')
    rollback_write_access_body = rollback[
        rollback_write_access_start : rollback.index("end)", rollback_write_access_start)
    ]

    assert "saveJournal" not in write_access_body
    assert "saveJournal" not in rollback_write_access_body
    assert write_chunk.index("saveJournal(journal)") > write_chunk.index("if not writeOk then")
    assert rollback.index("saveJournal(executionJournal)") > rollback.index("end)", rollback_write_access_start)


def test_plugin_writes_only_rating_metadata_and_never_removes_catalog_entries() -> None:
    source = all_plugin_source()

    assert 'setRawMetadata("rating"' in source
    for forbidden_key in (
        "pickStatus",
        "colorNameForLabel",
        "keywordTags",
        "developSettings",
        "removePhoto",
        "deletePhoto",
    ):
        assert forbidden_key not in source


def test_plugin_has_pending_rating_recovery_and_star_only_rollback() -> None:
    operation = plugin_file("CatalogOperation.lua")

    assert "pending_rating" in operation
    assert "manual_recovery_required" in operation
    assert "function CatalogOperation.rollback" in operation
    assert "rolled_back" in operation
