local LrTasks = import "LrTasks"
local PluginState = require "PluginState"

PluginState.running = true
PluginState.generation = PluginState.generation + 1
local generation = PluginState.generation
PluginState.started = true

LrTasks.startAsyncTask(function()
    while PluginState.running and PluginState.generation == generation do
        local workOk, workError = LrTasks.pcall(function()
            local loaded, Bridge = LrTasks.pcall(require, "Bridge")
            if not loaded then
                error(Bridge)
            end
            assert(Bridge and Bridge.processOnce, "bridge_process_missing")
            Bridge.processOnce()
        end)
        if workOk then
            PluginState.lastError = nil
        else
            PluginState.lastError = tostring(workError)
        end
        local heartbeatOk, heartbeatError = LrTasks.pcall(PluginState.writeHeartbeat)
        if not heartbeatOk then
            PluginState.lastError = tostring(heartbeatError)
        end
        LrTasks.sleep(2)
    end
    if PluginState.generation == generation then
        PluginState.started = false
    end
end)
