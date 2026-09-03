local LrDate = import "LrDate"
local LrFileUtils = import "LrFileUtils"
local LrPathUtils = import "LrPathUtils"
local LrTasks = import "LrTasks"
local LrUUID = import "LrUUID"
local Json = require "Json"

-- The background poller and the manual command can both publish a heartbeat.
-- Lightroom cannot replace an existing file atomically, so serialize writers
-- for the same outbox target before the delete-and-move publication step.
local outboxWriteLeases = {}

local PluginState = {
    running = false,
    started = false,
    generation = 0,
    version = "0.2.1",
    sdkVersion = 15.0,
    lastError = nil,
    lastHeartbeat = nil,
}

local directoryNames = {
    "inbox",
    "processing",
    "outbox",
    "archive",
    "quarantine",
    "journals",
    "operations",
}

local function acquireOutboxWrite(target)
    while outboxWriteLeases[target] do
        LrTasks.sleep(0.01)
    end
    outboxWriteLeases[target] = true
end

local function releaseOutboxWrite(target)
    outboxWriteLeases[target] = nil
end

function PluginState.bridgeRoot()
    local appData = LrPathUtils.getStandardFilePath("appData")
    return LrPathUtils.child(LrPathUtils.child(appData, "YidianPhotoCull"), "lightroom-bridge")
end

function PluginState.ensureBridgeDirectories()
    local root = PluginState.bridgeRoot()
    LrFileUtils.createAllDirectories(root)
    for _, name in ipairs(directoryNames) do
        LrFileUtils.createAllDirectories(LrPathUtils.child(root, name))
    end
    return root
end

function PluginState.writeAtomicOutbox(filename, payload)
    local root = PluginState.ensureBridgeDirectories()
    local outbox = LrPathUtils.child(root, "outbox")
    local target = LrPathUtils.child(outbox, filename)
    local temporary = target .. "." .. LrUUID.generateUUID() .. ".tmp"
    local stream = nil

    acquireOutboxWrite(target)
    local ok, result = LrTasks.pcall(function()
        local openError = nil
        stream, openError = io.open(temporary, "wb")
        if not stream then
            error(openError or "无法创建一点筛图临时收据")
        end
        stream:write(Json.encode(payload))
        stream:flush()
        stream:close()
        stream = nil
        if LrFileUtils.exists(target) then
            local deleted, deleteError = LrFileUtils.delete(target)
            if not deleted then
                error(deleteError or "无法替换一点筛图收据")
            end
        end
        local moved, moveError = LrFileUtils.move(temporary, target)
        if not moved then
            error(moveError or "无法原子发布一点筛图收据")
        end
        return target
    end)
    if stream then
        pcall(function()
            stream:close()
        end)
    end
    if not ok then
        LrTasks.pcall(function()
            if LrFileUtils.exists(temporary) then
                LrFileUtils.delete(temporary)
            end
        end)
    end
    releaseOutboxWrite(target)
    if not ok then
        error(result)
    end
    return result
end

function PluginState.writeHeartbeat()
    local timestamp = os.date(
        "!%Y-%m-%dT%H:%M:%SZ",
        math.floor(LrDate.timeToPosixDate(LrDate.currentTime()))
    )
    PluginState.writeAtomicOutbox("plugin-heartbeat.json", {
        schema_version = 1,
        plugin_id = "com.yidian.photocull.lightroom",
        plugin_version = PluginState.version,
        sdk_version = PluginState.sdkVersion,
        timestamp = timestamp,
        running = PluginState.running,
        last_error = PluginState.lastError,
    })
    PluginState.lastHeartbeat = timestamp
end

return PluginState
