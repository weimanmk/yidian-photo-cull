return {
    LrSdkVersion = 15.0,
    LrSdkMinimumVersion = 14.3,
    LrToolkitIdentifier = "com.yidian.photocull.lightroom",
    LrPluginName = "一点筛图",
    VERSION = { major = 0, minor = 2, revision = 1, build = 1 },
    LrInitPlugin = "InitPlugin.lua",
    LrForceInitPlugin = true,
    LrShutdownPlugin = "ShutdownPlugin.lua",
    LrLibraryMenuItems = {
        { title = "检查一点筛图任务", file = "ManualCheck.lua" },
    },
}
