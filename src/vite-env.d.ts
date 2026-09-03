/// <reference types="vite/client" />

interface Window {
  desktop?: {
    selectFolder: () => Promise<string | null>
    selectOutputFolder: () => Promise<string | null>
    revealFile: (fileId: string) => Promise<boolean>
    backendInfo: () => Promise<{ status: string; error: string; url: string; token: string }>
    restartBackend: () => Promise<boolean>
    lightroomStatus: () => Promise<{
      lightroom: { found: boolean; path: string | null; source: string | null }
      plugin: { installed: boolean; compatible: boolean; version: string | null; path: string }
    }>
    installLightroomPlugin: () => Promise<{ installed: boolean; version: string; path: string; backup: string | null }>
    launchLightroom: () => Promise<{ found: boolean; path: string; source: string }>
    openLightroomLogs: () => Promise<boolean>
    minimize: () => Promise<void>
    maximize: () => Promise<void>
    close: () => Promise<void>
    platform: string
  }
}
