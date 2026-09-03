const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('desktop', {
  selectFolder: () => ipcRenderer.invoke('dialog:select-folder'),
  selectOutputFolder: () => ipcRenderer.invoke('dialog:select-output-folder'),
  revealFile: (fileId) => ipcRenderer.invoke('file:reveal', fileId),
  backendInfo: () => ipcRenderer.invoke('backend:info'),
  restartBackend: () => ipcRenderer.invoke('backend:restart'),
  lightroomStatus: () => ipcRenderer.invoke('lightroom:status'),
  installLightroomPlugin: () => ipcRenderer.invoke('lightroom:install-plugin'),
  launchLightroom: () => ipcRenderer.invoke('lightroom:launch'),
  openLightroomLogs: () => ipcRenderer.invoke('lightroom:open-logs'),
  minimize: () => ipcRenderer.invoke('window:minimize'),
  maximize: () => ipcRenderer.invoke('window:maximize'),
  close: () => ipcRenderer.invoke('window:close'),
  platform: process.platform,
})
