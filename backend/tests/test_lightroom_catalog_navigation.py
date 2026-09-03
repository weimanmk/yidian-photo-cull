from __future__ import annotations

from tests.lightroom_fixtures import plugin_file
from tests.lightroom_lua_runtime import run_lightroom_lua


def _run_navigation_scenario(assertions: str) -> None:
    operation_source = plugin_file("CatalogOperation.lua")
    run_lightroom_lua(
        f"""
package.preload.CatalogOperation = assert(loadstring([====[
{operation_source}
]====], "@CatalogOperation.lua"))

local calls = {{}}
local applicationView = {{
    switchToModule = function(moduleName)
        calls.moduleName = moduleName
    end,
    gridView = function()
        calls.gridView = true
    end,
}}
local modules = {{
    LrApplication = {{}},
    LrApplicationView = applicationView,
    LrFileUtils = {{}},
    LrPathUtils = {{
        parent = function(path)
            local parents = {{
                ["C:/photos/a/a.jpg"] = "C:/photos/a",
                ["C:/photos/b/b.jpg"] = "C:/photos/b",
                ["C:/photos/a/c.jpg"] = "C:/photos/a",
            }}
            return parents[path]
        end,
    }},
    LrTasks = {{
        pcall = pcall,
        sleep = function(seconds)
            calls.sleepSeconds = seconds
            calls.sourceViewSettled = true
        end,
    }},
}}
function import(name)
    return assert(modules[name], "unexpected import: " .. tostring(name))
end
package.loaded.Receipt = {{}}

local folderA = {{ name = "A" }}
local folderB = {{ name = "B" }}
local photos = {{
    ["C:/photos/a/a.jpg"] = {{ getRawMetadata = function(_, key) assert(key == "path"); return "C:/photos/a/a.jpg" end }},
    ["C:/photos/b/b.jpg"] = {{ getRawMetadata = function(_, key) assert(key == "path"); return "C:/photos/b/b.jpg" end }},
    ["C:/photos/a/c.jpg"] = {{ getRawMetadata = function(_, key) assert(key == "path"); return "C:/photos/a/c.jpg" end }},
}}
local catalog = {{
    findPhotoByPath = function(_, path)
        return photos[path]
    end,
    getFolderByPath = function(_, path)
        if path == "C:/photos/a" then return folderA end
        if path == "C:/photos/b" then return folderB end
        return nil
    end,
    setActiveSources = function(_, sources)
        calls.activeSources = sources
    end,
    setSelectedPhotos = function(_, primary, others)
        assert(calls.sourceViewSettled == true, "source view must settle before selecting photos")
        calls.primary = primary
        calls.otherSelected = others
    end,
}}
local manifest = {{
    items = {{
        {{ source_path = "C:/photos/a/a.jpg" }},
        {{ source_path = "C:/photos/b/b.jpg" }},
        {{ source_path = "C:/photos/a/c.jpg" }},
    }},
}}

local CatalogOperation = require "CatalogOperation"
local focused = CatalogOperation.focusImportedPhotos(manifest, catalog)

{assertions}
"""
    )


def test_completed_batch_activates_each_distinct_source_folder() -> None:
    _run_navigation_scenario(
        """
assert(focused == true, "completed batch should be focused")
assert(calls.moduleName == "library", "Lightroom should switch to the Library module")
assert(calls.gridView == true, "Lightroom should switch to Grid view")
assert(calls.sleepSeconds >= 0.1, "Lightroom should wait for the source view to settle")
assert(#calls.activeSources == 2, "duplicate source folders should be collapsed")
assert(calls.activeSources[1] == folderA, "first source folder should keep manifest order")
assert(calls.activeSources[2] == folderB, "second distinct source folder should be active")
"""
    )


def test_completed_batch_selects_every_resolved_manifest_photo() -> None:
    _run_navigation_scenario(
        """
assert(calls.primary == photos["C:/photos/a/a.jpg"], "first photo should be most selected")
assert(#calls.otherSelected == 2, "remaining imported photos should also be selected")
assert(calls.otherSelected[1] == photos["C:/photos/b/b.jpg"], "selection should keep manifest order")
assert(calls.otherSelected[2] == photos["C:/photos/a/c.jpg"], "every resolved photo should be selected")
"""
    )


def test_successful_execute_focuses_the_completed_batch() -> None:
    operation_source = plugin_file("CatalogOperation.lua")
    run_lightroom_lua(
        f"""
package.preload.CatalogOperation = assert(loadstring([====[
{operation_source}
]====], "@CatalogOperation.lua"))

local calls = {{}}
local folder = {{ name = "event" }}
local photo = {{
    getRawMetadata = function(_, key)
        if key == "rating" then return 1 end
        if key == "path" then return "C:/photos/a.jpg" end
        error("unexpected metadata key: " .. tostring(key))
    end,
}}
local catalog = {{
    getPath = function() return "C:/catalog/test.lrcat" end,
    findPhotoByPath = function(_, path)
        if path == "C:/photos/a.jpg" then return photo end
        return nil
    end,
    getFolderByPath = function(_, path)
        if path == "C:/photos" then return folder end
        return nil
    end,
    setActiveSources = function(_, sources)
        calls.activeSources = sources
    end,
    setSelectedPhotos = function(_, primary, others)
        assert(calls.sourceViewSettled == true, "source view must settle before selecting photos")
        calls.primary = primary
        calls.others = others
    end,
}}
local receiptModule = {{
    catalogIdentity = function(path)
        assert(path == "C:/catalog/test.lrcat")
        return "catalog-id"
    end,
    forExecution = function()
        return {{
            counts = {{ new = 0, update = 0, verified = 0, pending_rating = 0 }},
            items = {{}},
            chunks = {{}},
        }}
    end,
    finishExecution = function(receipt, status)
        receipt.status = status
        return receipt
    end,
}}
local modules = {{
    LrApplication = {{ activeCatalog = function() return catalog end }},
    LrApplicationView = {{
        switchToModule = function(moduleName) calls.moduleName = moduleName end,
        gridView = function() calls.gridView = true end,
    }},
    LrFileUtils = {{
        exists = function() return "file" end,
        fileAttributes = function()
            return {{ fileSize = 10, fileModificationDate = 809776800 }}
        end,
    }},
    LrPathUtils = {{
        leafName = function() return "test.lrcat" end,
        parent = function(path)
            assert(path == "C:/photos/a.jpg")
            return "C:/photos"
        end,
    }},
        LrTasks = {{
            pcall = pcall,
            sleep = function(seconds)
                calls.sleepSeconds = seconds
                calls.sourceViewSettled = true
            end,
        }},
}}
function import(name)
    return assert(modules[name], "unexpected import: " .. tostring(name))
end
package.loaded.Receipt = receiptModule

local manifest = {{
    plan_hash = "plan",
    preflight_request_id = "preflight",
    baseline_hash = "baseline",
    catalog_identity_hash = "catalog-id",
    items = {{
        {{
            item_id = "photo-a",
            source_path = "C:/photos/a.jpg",
            path_hash = "path-a",
            file_size = 10,
            modified_ns = 1788084000000000000,
            target_rating = 1,
        }},
    }},
}}
local journal = {{
    plan_hash = "plan",
    preflight_request_id = "preflight",
    baseline_hash = "baseline",
    catalog_identity_hash = "catalog-id",
    progress = {{}},
    items = {{
        {{
            item_id = "photo-a",
            path_hash = "path-a",
            action = "unchanged",
            previous_rating = 1,
            target_rating = 1,
            status = "planned",
        }},
    }},
}}

local CatalogOperation = require "CatalogOperation"
local receipt = CatalogOperation.execute(manifest, journal, function()
    error("completed unchanged batch should not write a journal")
end)

assert(receipt.status == "complete", "fixture should finish successfully")
assert(calls.moduleName == "library", "successful execution should focus the imported batch")
assert(calls.activeSources[1] == folder, "successful execution should activate its source folder")
assert(calls.primary == photo, "successful execution should select its imported photo")
"""
    )
