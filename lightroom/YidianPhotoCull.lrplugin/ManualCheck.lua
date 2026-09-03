local LrTasks = import "LrTasks"
local PluginState = require "PluginState"

LrTasks.startAsyncTask(function()
    local loaded, Bridge = LrTasks.pcall(require, "Bridge")
    if not loaded then
        PluginState.lastError = tostring(Bridge)
    elseif Bridge and Bridge.processOnce then
        local ok, bridgeError = LrTasks.pcall(Bridge.processOnce)
        PluginState.lastError = ok and nil or tostring(bridgeError)
    else
        PluginState.lastError = "bridge_process_missing"
    end
    local ok, heartbeatError = LrTasks.pcall(PluginState.writeHeartbeat)
    if not ok then
        PluginState.lastError = tostring(heartbeatError)
    end
end)
