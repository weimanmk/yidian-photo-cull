local LrFileUtils = import "LrFileUtils"
local LrPathUtils = import "LrPathUtils"
local LrTasks = import "LrTasks"
local LrUUID = import "LrUUID"

local CatalogOperation = require "CatalogOperation"
local Json = require "Json"
local Receipt = require "Receipt"

local Bridge = {}

local knownOperations = {
    preflight = true,
    execute = true,
}

-- Background polling and the manual menu command can overlap. Keep one
-- in-memory lease per processing request; a plugin reload clears leases and
-- lets the durable processing queue resume normally.
local inFlightRequests = {}

local function requestKey(path)
    return LrPathUtils.leafName(path) or tostring(path)
end

local function requestInFlight(path)
    return inFlightRequests[requestKey(path)] == true
end

local function claimRequest(path)
    inFlightRequests[requestKey(path)] = true
end

local function releaseRequest(path)
    inFlightRequests[requestKey(path)] = nil
end

local function rootPath()
    local appData = LrPathUtils.getStandardFilePath("appData")
    return LrPathUtils.child(LrPathUtils.child(appData, "YidianPhotoCull"), "lightroom-bridge")
end

local function directory(name)
    local value = LrPathUtils.child(rootPath(), name)
    LrFileUtils.createAllDirectories(value)
    return value
end

local function readJson(path)
    local contents, readError = LrFileUtils.readFile(path)
    if not contents then
        error(readError or "无法读取一点筛图请求")
    end
    local decoded, _, decodeError = Json.decode(contents, 1, nil)
    if decodeError then
        error(decodeError)
    end
    return decoded
end

local function writeAtomicJson(directoryName, filename, payload)
    local parent = directory(directoryName)
    local target = LrPathUtils.child(parent, filename)
    local temporary = LrPathUtils.child(parent, "." .. LrUUID.generateUUID() .. ".tmp")
    local encoded = Json.encode(payload)
    if LrFileUtils.exists(target) == "file" then
        local existing = LrFileUtils.readFile(target)
        if existing == encoded then
            return target
        end
        error("immutable_file_conflict")
    end
    local stream, openError = io.open(temporary, "wb")
    if not stream then
        error(openError or "无法创建一点筛图临时文件")
    end
    local ok, writeError = LrTasks.pcall(function()
        stream:write(encoded)
        stream:flush()
        stream:close()
        local moved, moveError = LrFileUtils.move(temporary, target)
        if not moved then
            error(moveError or "无法原子发布一点筛图文件")
        end
    end)
    if not ok then
        pcall(function()
            stream:close()
        end)
        if LrFileUtils.exists(temporary) then
            LrFileUtils.delete(temporary)
        end
        error(writeError)
    end
    return target
end

local function validatePolicy(policy)
    return type(policy) == "table"
        and policy.import_mode == "in_place"
        and policy.collection_mode == "none"
        and policy.write_xmp == false
        and tonumber(policy.protect_existing_rating_above) == 3
end

local function validateManifest(manifest)
    if type(manifest) ~= "table" or tonumber(manifest.schema_version) ~= 1 then
        error("unsupported_schema")
    end
    if not knownOperations[manifest.operation] then
        error("unknown_operation")
    end
    if not validatePolicy(manifest.policy) then
        error("unsafe_policy")
    end
    if type(manifest.plan_hash) ~= "string" or not manifest.plan_hash:match("^[0-9a-f][0-9a-f]+$") or #manifest.plan_hash ~= 64 then
        error("invalid_plan_hash")
    end
    if type(manifest.items) ~= "table" or #manifest.items == 0 then
        error("empty_plan")
    end
    if manifest.operation == "execute" then
        if type(manifest.preflight_request_id) ~= "string"
            or type(manifest.baseline_hash) ~= "string"
            or #manifest.baseline_hash ~= 64
            or type(manifest.catalog_identity_hash) ~= "string"
            or #manifest.catalog_identity_hash ~= 64 then
            error("invalid_execute_baseline")
        end
    end
end

local function quarantineCandidate(candidate, manifest, reason)
    local leaf = LrPathUtils.leafName(candidate)
    local target = LrPathUtils.child(directory("quarantine"), leaf)
    LrFileUtils.move(candidate, target)
    if type(manifest) == "table" then
        local receipt = Receipt.error(manifest, reason, "invalid_request")
        writeAtomicJson("outbox", tostring(receipt.request_id) .. ".json", receipt)
    end
end

function Bridge.claimNext(expectedOperation)
    local function candidatesIn(directoryName)
        local candidates = {}
        for path in LrFileUtils.files(directory(directoryName)) do
            if LrPathUtils.extension(path) == "json" then
                table.insert(candidates, path)
            end
        end
        table.sort(candidates)
        return candidates
    end

    local function quarantineClaimed(candidate, manifest, reason)
        local quarantined, quarantineError = LrTasks.pcall(quarantineCandidate, candidate, manifest, reason)
        releaseRequest(candidate)
        if not quarantined then
            error(quarantineError)
        end
    end

    for _, candidate in ipairs(candidatesIn("processing")) do
        if not requestInFlight(candidate) then
            claimRequest(candidate)
            local decoded, manifest = LrTasks.pcall(readJson, candidate)
            if not decoded then
                quarantineClaimed(candidate, nil, manifest)
            else
                local valid, validationError = pcall(validateManifest, manifest)
                if not valid then
                    quarantineClaimed(candidate, manifest, validationError)
                elseif expectedOperation == nil or manifest.operation == expectedOperation then
                    return candidate, manifest
                else
                    releaseRequest(candidate)
                end
            end
        end
    end

    for _, candidate in ipairs(candidatesIn("inbox")) do
        if not requestInFlight(candidate) then
            claimRequest(candidate)
            local decoded, manifest = LrTasks.pcall(readJson, candidate)
            if not decoded then
                quarantineClaimed(candidate, nil, manifest)
            else
                local valid, validationError = pcall(validateManifest, manifest)
                if not valid then
                    quarantineClaimed(candidate, manifest, validationError)
                elseif expectedOperation == nil or manifest.operation == expectedOperation then
                    local target = LrPathUtils.child(directory("processing"), LrPathUtils.leafName(candidate))
                    local moved = LrFileUtils.move(candidate, target)
                    if moved then
                        return target, manifest
                    end
                    releaseRequest(candidate)
                else
                    releaseRequest(candidate)
                end
            end
        end
    end
    return nil, nil
end

local function serializable(value)
    if type(value) ~= "table" then
        return value
    end
    local copied = {}
    for key, child in pairs(value) do
        if type(child) ~= "function" and tostring(key):sub(1, 1) ~= "_" then
            copied[key] = serializable(child)
        end
    end
    return copied
end

function Bridge.writeJournal(journal)
    journal.revision = tonumber(journal.revision or 0) + 1
    local filename = string.format("%s.%08d.json", tostring(journal.operation_id), journal.revision)
    writeAtomicJson("journals", filename, serializable(journal))
    return journal
end

function Bridge.readLatestJournal(operationId)
    local selectedPath = nil
    local selectedRevision = -1
    local prefix = "^" .. tostring(operationId) .. "%.(%d+)%.json$"
    for path in LrFileUtils.files(directory("journals")) do
        local revision = tonumber(LrPathUtils.leafName(path):match(prefix))
        if revision and revision > selectedRevision then
            selectedPath = path
            selectedRevision = revision
        end
    end
    if not selectedPath then
        return nil
    end
    return readJson(selectedPath)
end

function Bridge.writePreflightJournal(manifest, receipt)
    local journal = {
        schema_version = 1,
        operation_id = manifest.operation_id,
        preflight_request_id = manifest.request_id,
        plan_hash = manifest.plan_hash,
        baseline_hash = receipt.baseline_hash,
        catalog_name = receipt.catalog_name,
        catalog_identity_hash = receipt.catalog_identity_hash,
        counts = receipt.counts,
        items = receipt.items,
        progress = {},
        chunks = {},
        revision = -1,
    }
    return Bridge.writeJournal(journal)
end

function Bridge.complete(requestPath, receipt)
    local receiptPath = writeAtomicJson("outbox", tostring(receipt.request_id) .. ".json", receipt)
    local archivePath = LrPathUtils.child(directory("archive"), LrPathUtils.leafName(requestPath))
    if LrFileUtils.exists(archivePath) then
        archivePath = LrFileUtils.chooseUniqueFileName(archivePath)
    end
    local moved, moveError = LrFileUtils.move(requestPath, archivePath)
    if not moved then
        error(moveError or "无法归档一点筛图请求")
    end
    return receiptPath
end

function Bridge.reject(requestPath, manifest, reason)
    local receipt = Receipt.error(manifest, reason, "request_quarantined")
    writeAtomicJson("outbox", tostring(receipt.request_id) .. ".json", receipt)
    local quarantinePath = LrPathUtils.child(directory("quarantine"), LrPathUtils.leafName(requestPath))
    LrFileUtils.move(requestPath, quarantinePath)
    return receipt
end

function Bridge.processOnce()
    local requestPath, manifest = Bridge.claimNext(nil)
    if not requestPath then
        return false
    end
    local ok, result = LrTasks.pcall(function()
        local receipt = nil
        if manifest.operation == "preflight" then
            receipt = CatalogOperation.preflight(manifest)
            Bridge.writePreflightJournal(manifest, receipt)
        elseif manifest.operation == "execute" then
            local journal = Bridge.readLatestJournal(manifest.operation_id)
            if not journal
                or journal.plan_hash ~= manifest.plan_hash
                or journal.preflight_request_id ~= manifest.preflight_request_id
                or journal.baseline_hash ~= manifest.baseline_hash
                or journal.catalog_identity_hash ~= manifest.catalog_identity_hash then
                error("preflight_journal_mismatch")
            end
            receipt = CatalogOperation.execute(manifest, journal, Bridge.writeJournal)
        else
            error("unknown_operation")
        end
        Bridge.complete(requestPath, receipt)
        return receipt
    end)
    if not ok then
        local rejected, rejectError = LrTasks.pcall(Bridge.reject, requestPath, manifest, result)
        releaseRequest(requestPath)
        if not rejected then
            error(rejectError)
        end
        return true
    end
    releaseRequest(requestPath)
    return true
end

return Bridge
