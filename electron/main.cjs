const { app, BrowserWindow, dialog, ipcMain, shell } = require('electron')
const { spawn } = require('node:child_process')
const crypto = require('node:crypto')
const fs = require('node:fs')
const path = require('node:path')
const http = require('node:http')
const { createLightroomService } = require('./lightroom-service.cjs')

const BACKEND_PORT = 8767
const BACKEND_URL = `http://127.0.0.1:${BACKEND_PORT}`
const BACKEND_TOKEN = crypto.randomBytes(32).toString('hex')
const isDev = !app.isPackaged
const PRODUCT_NAME = '一点筛图'

app.setName(PRODUCT_NAME)

let mainWindow = null
let backendProcess = null
let backendState = { status: 'starting', error: '' }
let lightroomService = null

app.commandLine.appendSwitch('js-flags', '--max-old-space-size=4096')
app.commandLine.appendSwitch('disable-gpu-sandbox')

function backendCommand() {
  if (app.isPackaged) {
    const executable = path.join(process.resourcesPath, 'backend', 'photocull-backend.exe')
    return fs.existsSync(executable) ? { command: executable, args: [], cwd: path.dirname(executable) } : null
  }

  const root = path.resolve(__dirname, '..')
  const backendDir = path.join(root, 'backend')
  const configured = process.env.PHOTOCULL_PYTHON
  const candidates = [
    configured,
    path.join(root, '.venv-cuda312', 'Scripts', 'python.exe'),
    path.join(root, '.venv', 'Scripts', 'python.exe'),
    path.join(root, '.venv', 'bin', 'python'),
    'python',
  ].filter(Boolean)

  const command = candidates.find((candidate) => candidate === 'python' || fs.existsSync(candidate))
  return command ? { command, args: ['-m', 'photocull'], cwd: backendDir } : null
}

function startBackend() {
  if (backendProcess) return
  const spec = backendCommand()
  if (!spec) {
    backendState = { status: 'failed', error: '未找到本地 AI 后端。请先创建 Python 3.12 虚拟环境并安装 backend。' }
    return
  }

  backendState = { status: 'starting', error: '' }
  backendProcess = spawn(spec.command, spec.args, {
    cwd: spec.cwd,
    windowsHide: true,
    env: {
      ...process.env,
      PYTHONUTF8: '1',
      PYTHONDONTWRITEBYTECODE: '1',
      PHOTOCULL_PORT: String(BACKEND_PORT),
      PHOTOCULL_API_TOKEN: BACKEND_TOKEN,
      PHOTOCULL_DISABLE_VLM: app.isPackaged ? '1' : (process.env.PHOTOCULL_DISABLE_VLM || '0'),
      PHOTOCULL_MODEL_DIR: app.isPackaged ? path.join(process.resourcesPath, 'models') : (process.env.PHOTOCULL_MODEL_DIR || ''),
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  })

  backendProcess.stdout.on('data', (chunk) => {
    const message = chunk.toString()
    if (/Uvicorn running|Application startup complete|启动完成/.test(message)) {
      backendState = { status: 'running', error: '' }
    }
    if (isDev) process.stdout.write(`[AI] ${message}`)
  })
  backendProcess.stderr.on('data', (chunk) => {
    const message = chunk.toString()
    if (isDev) process.stderr.write(`[AI] ${message}`)
    backendState.error = `${backendState.error}${message}`.slice(-3000)
  })
  backendProcess.on('error', (error) => {
    backendState = { status: 'failed', error: error.message }
    backendProcess = null
  })
  backendProcess.on('exit', (code) => {
    if (code && code !== 0) backendState = { status: 'failed', error: backendState.error || `后端退出，代码 ${code}` }
    backendProcess = null
  })
}

function requestBackend(pathname, method = 'GET') {
  return new Promise((resolve) => {
    const request = http.request(`${BACKEND_URL}${pathname}`, { method, timeout: 1200, headers: { 'X-PhotoCull-Token': BACKEND_TOKEN } }, (response) => {
      response.resume()
      response.on('end', () => resolve(response.statusCode >= 200 && response.statusCode < 300))
    })
    request.on('timeout', () => {
      request.destroy()
      resolve(false)
    })
    request.on('error', () => resolve(false))
    request.end()
  })
}

async function stopBackend() {
  if (!backendProcess) return
  await requestBackend('/api/shutdown', 'POST')
  const processToStop = backendProcess
  setTimeout(() => {
    if (backendProcess === processToStop) processToStop.kill()
  }, 1800).unref()
}

function getLightroomService() {
  if (!lightroomService) {
    lightroomService = createLightroomService({
      appData: app.getPath('appData'),
      resources: process.resourcesPath,
      projectRoot: path.resolve(__dirname, '..'),
    })
  }
  return lightroomService
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1500,
    height: 940,
    minWidth: 1120,
    minHeight: 720,
    frame: false,
    // 应用默认是白天模式；在 React 首屏完成前也使用同色背景，避免启动时出现大块黑屏。
    backgroundColor: '#f3f6f8',
    title: PRODUCT_NAME,
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  })

  if (isDev) {
    mainWindow.loadURL('http://127.0.0.1:5173')
  } else {
    mainWindow.loadFile(path.join(__dirname, '..', 'dist', 'index.html'))
  }
  mainWindow.on('closed', () => { mainWindow = null })
}

app.whenReady().then(() => {
  app.setAppUserModelId('com.yidian.photocull')
  startBackend()
  createWindow()
})

app.on('window-all-closed', async () => {
  await stopBackend()
  app.quit()
})

app.on('before-quit', () => { stopBackend() })

ipcMain.handle('dialog:select-folder', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: '选择待筛选照片文件夹',
    properties: ['openDirectory'],
  })
  return result.canceled ? null : result.filePaths[0]
})

ipcMain.handle('dialog:select-output-folder', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: '选择导出文件夹',
    properties: ['openDirectory', 'createDirectory'],
  })
  return result.canceled ? null : result.filePaths[0]
})

ipcMain.handle('file:reveal', async (_event, fileId) => {
  try {
    const response = await fetch(`${BACKEND_URL}/api/files/${encodeURIComponent(fileId)}`, { headers: { 'X-PhotoCull-Token': BACKEND_TOKEN } })
    if (!response.ok) return false
    const payload = await response.json()
    if (!payload.path || !fs.existsSync(payload.path)) return false
    shell.showItemInFolder(payload.path)
    return true
  } catch {
    return false
  }
})

ipcMain.handle('backend:info', async () => {
  const healthy = await requestBackend('/api/health')
  if (healthy) backendState = { status: 'running', error: '' }
  return { ...backendState, url: BACKEND_URL, token: BACKEND_TOKEN }
})

ipcMain.handle('backend:restart', async () => {
  await stopBackend()
  setTimeout(startBackend, 800)
  return true
})

ipcMain.handle('lightroom:status', async () => {
  const service = getLightroomService()
  const [lightroom, plugin] = await Promise.all([service.detectLightroom(), service.pluginStatus()])
  return { lightroom, plugin }
})

ipcMain.handle('lightroom:install-plugin', async () => getLightroomService().installPlugin())
ipcMain.handle('lightroom:launch', async () => getLightroomService().launchLightroom())
ipcMain.handle('lightroom:open-logs', async () => {
  const target = await getLightroomService().logsPath()
  return (await shell.openPath(target)) === ''
})

ipcMain.handle('window:minimize', () => mainWindow?.minimize())
ipcMain.handle('window:maximize', () => mainWindow?.isMaximized() ? mainWindow.unmaximize() : mainWindow?.maximize())
ipcMain.handle('window:close', () => mainWindow?.close())
