local LrDate = import "LrDate"
local LrDigest = import "LrDigest"
local Json = require "Json"

local Receipt = {}

local function timestamp()
    return os.date(
        "!%Y-%m-%dT%H:%M:%SZ",
        math.floor(LrDate.timeToPosixDate(LrDate.currentTime()))
    )
end

local function digest64(value)
    local text = tostring(value or "")
    return LrDigest.SHA256.digest(text)
end

function Receipt.timestamp()
    return timestamp()
end

local function emptyCounts(total)
    return {
        total = total or 0,
        new = 0,
        update = 0,
        unchanged = 0,
        protected = 0,
        invalid = 0,
        catalog_added = 0,
        pending_rating = 0,
        verified = 0,
        rolled_back = 0,
    }
end

function Receipt.catalogIdentity(catalogPath)
    return digest64(catalogPath)
end

function Receipt.begin(manifest, catalogName, catalogIdentity)
    return {
        schema_version = 1,
        request_id = manifest.request_id,
        operation_id = manifest.operation_id,
        plan_hash = manifest.plan_hash,
        catalog_name = catalogName,
        catalog_identity_hash = catalogIdentity,
        started_at = timestamp(),
        finished_at = timestamp(),
        status = "awaiting_confirmation",
        counts = emptyCounts(#manifest.items),
        chunks = {},
        items = {},
    }
end

function Receipt.addItem(receipt, plannedItem, action, previousRating, finalRating, status)
    local item = {
        item_id = plannedItem.item_id,
        path_hash = plannedItem.path_hash,
        action = action,
        target_rating = plannedItem.target_rating,
        status = status,
    }
    if previousRating ~= nil then
        item.previous_rating = previousRating
    end
    if finalRating ~= nil then
        item.final_rating = finalRating
    end
    table.insert(receipt.items, item)
    receipt.counts[action] = receipt.counts[action] + 1
    return item
end

function Receipt.baselineHash(items)
    local rows = {}
    for _, item in ipairs(items) do
        table.insert(rows, {
            item_id = item.item_id,
            path_hash = item.path_hash,
            action = item.action,
            previous_rating = item.previous_rating,
            target_rating = item.target_rating,
            status = item.status,
        })
    end
    table.sort(rows, function(left, right)
        return left.item_id < right.item_id
    end)
    return digest64(Json.encode(rows))
end

function Receipt.finishPreflight(receipt)
    receipt.baseline_hash = Receipt.baselineHash(receipt.items)
    receipt.finished_at = timestamp()
    return receipt
end

local function copyCounts(counts)
    return {
        total = counts.total or 0,
        new = counts.new or 0,
        update = counts.update or 0,
        unchanged = counts.unchanged or 0,
        protected = counts.protected or 0,
        invalid = counts.invalid or 0,
        catalog_added = 0,
        pending_rating = 0,
        verified = 0,
        rolled_back = 0,
    }
end

local function copyReceiptItem(item)
    local copied = {
        item_id = item.item_id,
        path_hash = item.path_hash,
        action = item.action,
        target_rating = item.target_rating,
        status = item.status,
    }
    if item.previous_rating ~= nil then
        copied.previous_rating = item.previous_rating
    end
    if item.final_rating ~= nil then
        copied.final_rating = item.final_rating
    end
    return copied
end

function Receipt.forExecution(manifest, journal)
    local receipt = {
        schema_version = 1,
        request_id = manifest.request_id,
        operation_id = manifest.operation_id,
        plan_hash = manifest.plan_hash,
        baseline_hash = journal.baseline_hash,
        catalog_name = journal.catalog_name,
        catalog_identity_hash = journal.catalog_identity_hash,
        started_at = timestamp(),
        finished_at = timestamp(),
        status = "pending_rating",
        counts = copyCounts(journal.counts),
        chunks = {},
        items = {},
    }
    local progress = journal.progress or {}
    for _, item in ipairs(journal.items) do
        local copied = copyReceiptItem(item)
        local itemProgress = progress[item.item_id] or {}
        if itemProgress.catalog_added then
            receipt.counts.catalog_added = receipt.counts.catalog_added + 1
        end
        if itemProgress.verified then
            receipt.counts.verified = receipt.counts.verified + 1
            copied.final_rating = item.target_rating
            copied.status = "verified"
        elseif itemProgress.pending_rating then
            receipt.counts.pending_rating = receipt.counts.pending_rating + 1
            copied.status = "pending_rating"
        elseif item.action == "unchanged" or item.action == "protected" then
            copied.final_rating = item.previous_rating
        end
        table.insert(receipt.items, copied)
    end
    return receipt
end

function Receipt.addChunk(receipt, chunkId, status, counts, errorCode)
    local chunk = {
        chunk_id = chunkId,
        status = status,
        counts = counts or {},
    }
    if errorCode then
        chunk.error_code = errorCode
    end
    table.insert(receipt.chunks, chunk)
end

function Receipt.finishExecution(receipt, status, errorCode, errorMessage)
    receipt.status = status
    receipt.finished_at = timestamp()
    if errorCode then
        receipt.error_code = errorCode
    end
    if errorMessage then
        receipt.error_message = tostring(errorMessage)
    end
    return receipt
end

function Receipt.error(manifest, message, code)
    local fallbackHash = manifest.plan_hash or string.rep("0", 64)
    return {
        schema_version = 1,
        request_id = manifest.request_id or string.rep("0", 32),
        operation_id = manifest.operation_id or string.rep("0", 32),
        plan_hash = fallbackHash,
        catalog_name = "当前 Lightroom 目录",
        catalog_identity_hash = fallbackHash,
        started_at = timestamp(),
        finished_at = timestamp(),
        status = "quarantined",
        counts = emptyCounts(0),
        chunks = {},
        items = {},
        error_code = code or "invalid_request",
        error_message = tostring(message),
    }
end

return Receipt
