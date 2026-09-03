local LrApplication = import "LrApplication"
local LrApplicationView = import "LrApplicationView"
local LrFileUtils = import "LrFileUtils"
local LrPathUtils = import "LrPathUtils"
local LrTasks = import "LrTasks"

local Receipt = require "Receipt"

local CatalogOperation = {}

local UPDATE_CHUNK_SIZE = 100
local NEW_CHUNK_SIZE = 1
local LIGHTROOM_EPOCH_TO_UNIX_SECONDS = 978307200

local function normalizedRating(value)
    return tonumber(value) or 0
end

local function ratingValue(target)
    if tonumber(target) == 0 then
        return nil
    end
    return tonumber(target)
end

local function fingerprintMatches(item)
    if LrFileUtils.exists(item.source_path) ~= "file" then
        return false
    end
    local attributes = LrFileUtils.fileAttributes(item.source_path)
    if not attributes then
        return false
    end
    local actualSize = tonumber(attributes.fileSize)
    local lightroomSeconds = tonumber(attributes.fileModificationDate)
    local actualUnixSeconds = lightroomSeconds and math.floor(lightroomSeconds + LIGHTROOM_EPOCH_TO_UNIX_SECONDS) or nil
    local plannedSeconds = math.floor(tonumber(item.modified_ns) / 1000000000)
    return actualSize == tonumber(item.file_size) and actualUnixSeconds == plannedSeconds
end

CatalogOperation.fingerprintMatches = fingerprintMatches

function CatalogOperation.focusImportedPhotos(manifest, catalog)
    local photos = {}
    local folders = {}
    local seenFolders = {}
    for _, item in ipairs(manifest.items or {}) do
        local photo = catalog:findPhotoByPath(item.source_path)
        if photo then
            table.insert(photos, photo)
        end
        local folder = nil
        if photo then
            local photoPath = photo:getRawMetadata("path")
            local folderPath = photoPath and LrPathUtils.parent(photoPath) or nil
            folder = folderPath and catalog:getFolderByPath(folderPath) or nil
        end
        if folder and not seenFolders[folder] then
            seenFolders[folder] = true
            table.insert(folders, folder)
        end
    end
    if #folders == 0 then
        return false
    end
    local focused = LrTasks.pcall(function()
        LrApplicationView.switchToModule("library")
        LrApplicationView.gridView()
        catalog:setActiveSources(folders)
        LrTasks.sleep(0.25)
        local otherSelected = {}
        for index = 2, #photos do
            table.insert(otherSelected, photos[index])
        end
        catalog:setSelectedPhotos(photos[1], otherSelected)
    end)
    return focused == true
end

local function classify(catalog, item)
    if not fingerprintMatches(item) then
        return "invalid", nil, "invalid"
    end
    local photo = catalog:findPhotoByPath(item.source_path)
    if not photo then
        return "new", nil, "planned"
    end
    local currentRating = normalizedRating(photo:getRawMetadata("rating"))
    if currentRating > 3 then
        return "protected", currentRating, "protected"
    end
    if currentRating == tonumber(item.target_rating) then
        return "unchanged", currentRating, "planned"
    end
    return "update", currentRating, "planned"
end

local function activeCatalogDetails()
    local catalog = LrApplication.activeCatalog()
    local catalogPath = catalog:getPath()
    local catalogName = LrPathUtils.leafName(catalogPath) or "当前 Lightroom 目录"
    return catalog, catalogPath, catalogName
end

function CatalogOperation.preflight(manifest)
    local catalog, catalogPath, catalogName = activeCatalogDetails()
    local receipt = Receipt.begin(manifest, catalogName, Receipt.catalogIdentity(catalogPath))
    for _, item in ipairs(manifest.items) do
        local action, previousRating, status = classify(catalog, item)
        Receipt.addItem(receipt, item, action, previousRating, nil, status)
    end
    return Receipt.finishPreflight(receipt)
end

local function indexById(items)
    local indexed = {}
    for _, item in ipairs(items or {}) do
        indexed[item.item_id] = item
    end
    return indexed
end

local function progressFor(journal, itemId)
    journal.progress = journal.progress or {}
    journal.progress[itemId] = journal.progress[itemId] or {}
    return journal.progress[itemId]
end

function CatalogOperation.revalidateBaseline(manifest, journal)
    local catalog, catalogPath = activeCatalogDetails()
    if manifest.plan_hash ~= journal.plan_hash
        or manifest.preflight_request_id ~= journal.preflight_request_id
        or manifest.baseline_hash ~= journal.baseline_hash
        or manifest.catalog_identity_hash ~= Receipt.catalogIdentity(catalogPath)
        or manifest.catalog_identity_hash ~= journal.catalog_identity_hash then
        return false, "catalog_or_plan_drift"
    end
    local plannedItems = indexById(manifest.items)
    local progress = journal.progress or {}
    for _, baseline in ipairs(journal.items or {}) do
        local item = plannedItems[baseline.item_id]
        if not item or item.path_hash ~= baseline.path_hash or item.target_rating ~= baseline.target_rating then
            return false, "item_plan_drift"
        end
        if not fingerprintMatches(item) then
            return false, "source_fingerprint_drift"
        end
        local photo = catalog:findPhotoByPath(item.source_path)
        local itemProgress = progress[item.item_id] or {}
        if baseline.action == "new" then
            if photo and not itemProgress.catalog_added then
                return false, "catalog_membership_drift"
            end
            if photo and itemProgress.catalog_added and itemProgress.verified
                and normalizedRating(photo:getRawMetadata("rating")) ~= tonumber(item.target_rating) then
                return false, "rating_drift"
            end
        elseif baseline.action == "invalid" then
            return false, "invalid_preflight_item"
        else
            if not photo then
                return false, "catalog_membership_drift"
            end
            local currentRating = normalizedRating(photo:getRawMetadata("rating"))
            if itemProgress.verified then
                if currentRating ~= tonumber(item.target_rating) then
                    return false, "rating_drift"
                end
            elseif currentRating ~= normalizedRating(baseline.previous_rating) then
                return false, "rating_drift"
            end
        end
    end
    return true, nil
end

local function pendingItems(manifest, journal, action)
    local manifestItems = indexById(manifest.items)
    local selected = {}
    for _, baseline in ipairs(journal.items) do
        local progress = progressFor(journal, baseline.item_id)
        if baseline.action == action and not progress.verified then
            table.insert(selected, {
                planned = manifestItems[baseline.item_id],
                baseline = baseline,
                progress = progress,
            })
        end
    end
    return selected
end

local function chunks(items, size)
    local result = {}
    local current = nil
    for index, item in ipairs(items) do
        if (index - 1) % size == 0 then
            current = {}
            table.insert(result, current)
        end
        table.insert(current, item)
    end
    return result
end

local function verifyChunk(catalog, entries)
    local verified = 0
    local pending = 0
    for _, entry in ipairs(entries) do
        local item = entry.planned
        local photo = catalog:findPhotoByPath(item.source_path)
        if photo and normalizedRating(photo:getRawMetadata("rating")) == tonumber(item.target_rating) then
            entry.progress.verified = true
            entry.progress.pending_rating = false
            verified = verified + 1
        else
            entry.progress.verified = false
            entry.progress.pending_rating = entry.progress.catalog_added == true
            if entry.progress.pending_rating then
                pending = pending + 1
            end
        end
    end
    return verified, pending
end

local function restoreUpdateChunk(catalog, entries)
    local writeOk = LrTasks.pcall(function()
        catalog:withWriteAccessDo("一点筛图 · 恢复原星级", function()
            for _, entry in ipairs(entries) do
                local photo = catalog:findPhotoByPath(entry.planned.source_path)
                if photo then
                    photo:setRawMetadata("rating", ratingValue(entry.baseline.previous_rating))
                end
            end
        end)
    end)
    if not writeOk then
        return false
    end
    local restored = true
    for _, entry in ipairs(entries) do
        local photo = catalog:findPhotoByPath(entry.planned.source_path)
        if not photo or normalizedRating(photo:getRawMetadata("rating")) ~= normalizedRating(entry.baseline.previous_rating) then
            restored = false
        end
        entry.progress.verified = false
    end
    return restored
end

local function writeChunk(catalog, entries, kind, journal, saveJournal)
    local catalogProgressChanged = false
    local writeOk, writeError = LrTasks.pcall(function()
        catalog:withWriteAccessDo("一点筛图 · 写入星级", function()
            for _, entry in ipairs(entries) do
                local item = entry.planned
                local photo = catalog:findPhotoByPath(item.source_path)
                if kind == "new" and not photo then
                    photo = assert(catalog:addPhoto(item.source_path))
                    entry.progress.catalog_added = true
                    catalogProgressChanged = true
                end
                if not photo then
                    error("catalog_photo_missing")
                end
                photo:setRawMetadata("rating", ratingValue(item.target_rating))
            end
        end)
    end)
    if not writeOk then
        if catalogProgressChanged then
            saveJournal(journal)
            catalogProgressChanged = false
        end
        if kind == "update" and not restoreUpdateChunk(catalog, entries) then
            return false, "manual_recovery_required", writeError
        end
        return false, kind == "new" and "pending_rating" or "failed", writeError
    end
    if catalogProgressChanged then
        saveJournal(journal)
    end
    local verified, pending = verifyChunk(catalog, entries)
    saveJournal(journal)
    if verified ~= #entries then
        if kind == "update" and not restoreUpdateChunk(catalog, entries) then
            return false, "manual_recovery_required", "update_readback_failed"
        end
        return false, pending > 0 and "pending_rating" or "failed", "readback_failed"
    end
    return true, "complete", nil
end

local function refreshReceipt(manifest, journal)
    return Receipt.forExecution(manifest, journal)
end

function CatalogOperation.execute(manifest, preflightJournal, saveJournal)
    local baselineOk, baselineError = CatalogOperation.revalidateBaseline(manifest, preflightJournal)
    if not baselineOk then
        local driftReceipt = refreshReceipt(manifest, preflightJournal)
        return Receipt.finishExecution(driftReceipt, "replan_required", baselineError, "Lightroom 目录基线已变化")
    end
    local catalog = LrApplication.activeCatalog()
    local work = {
        { action = "update", size = UPDATE_CHUNK_SIZE },
        { action = "new", size = NEW_CHUNK_SIZE },
    }
    for _, kind in ipairs(work) do
        local actionChunks = chunks(pendingItems(manifest, preflightJournal, kind.action), kind.size)
        for chunkIndex, entries in ipairs(actionChunks) do
            local currentOk, currentError = CatalogOperation.revalidateBaseline(manifest, preflightJournal)
            if not currentOk then
                local driftReceipt = refreshReceipt(manifest, preflightJournal)
                return Receipt.finishExecution(driftReceipt, "replan_required", currentError, "写入前目录基线已变化")
            end
            local chunkId = string.format("%s-%04d", kind.action, chunkIndex)
            local ok, status, errorMessage = writeChunk(catalog, entries, kind.action, preflightJournal, saveJournal)
            local receipt = refreshReceipt(manifest, preflightJournal)
            Receipt.addChunk(receipt, chunkId, status, { total = #entries }, ok and nil or status)
            if not ok then
                return Receipt.finishExecution(receipt, status, status, errorMessage)
            end
        end
    end
    local receipt = refreshReceipt(manifest, preflightJournal)
    if receipt.counts.verified == receipt.counts.new + receipt.counts.update
        and receipt.counts.pending_rating == 0 then
        local completed = Receipt.finishExecution(receipt, "complete")
        CatalogOperation.focusImportedPhotos(manifest, catalog)
        return completed
    end
    return Receipt.finishExecution(receipt, "pending_rating", "pending_rating", "仍有星级等待权威回读")
end

function CatalogOperation.rollback(manifest, executionJournal, saveJournal)
    local catalog = LrApplication.activeCatalog()
    local manifestItems = indexById(manifest.items)
    local receipt = Receipt.forExecution(manifest, executionJournal)
    receipt.status = "rollback_awaiting_confirmation"
    receipt.counts.rolled_back = 0
    local conflicts = 0
    local journalChanged = false
    catalog:withWriteAccessDo("一点筛图 · 撤销本次星级", function()
        for _, baseline in ipairs(executionJournal.items or {}) do
            local progress = (executionJournal.progress or {})[baseline.item_id] or {}
            if progress.verified and (baseline.action == "new" or baseline.action == "update") then
                local item = manifestItems[baseline.item_id]
                local photo = item and catalog:findPhotoByPath(item.source_path) or nil
                if photo and normalizedRating(photo:getRawMetadata("rating")) == tonumber(item.target_rating) then
                    local restoredRating = baseline.action == "new" and nil or ratingValue(baseline.previous_rating)
                    photo:setRawMetadata("rating", restoredRating)
                    progress.rolled_back = true
                    receipt.counts.rolled_back = receipt.counts.rolled_back + 1
                    journalChanged = true
                else
                    conflicts = conflicts + 1
                end
            end
        end
    end)
    if journalChanged then
        saveJournal(executionJournal)
    end
    for _, item in ipairs(receipt.items) do
        local progress = (executionJournal.progress or {})[item.item_id] or {}
        if progress.rolled_back then
            item.action = "rollback"
            item.final_rating = item.previous_rating or 0
            item.status = "rolled_back"
        end
    end
    if conflicts > 0 then
        return Receipt.finishExecution(receipt, "rolled_back", "rollback_conflicts", "部分照片已被用户修改，已跳过")
    end
    return Receipt.finishExecution(receipt, "rolled_back")
end

return CatalogOperation
