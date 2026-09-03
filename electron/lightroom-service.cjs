const childProcess = require('node:child_process')
const crypto = require('node:crypto')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const { promisify } = require('node:util')

const PLUGIN_ID = 'com.yidian.photocull.lightroom'
const PLUGIN_VERSION = '0.2.1'
const PLUGIN_DIRECTORY = 'YidianPhotoCull.lrplugin'
const APP_PATHS_KEY = 'HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\App Paths\\Lightroom.exe'

function isInside(root, candidate) {
  const relative = path.relative(root, candidate)
  return relative !== '' && !relative.startsWith('..') && !path.isAbsolute(relative)
}

async function regularFile(filePath) {
  try {
    return (await fs.promises.lstat(filePath)).isFile()
  } catch {
    return false
  }
}

function safeManifestPath(root, value) {
  if (typeof value !== 'string' || !value.trim()) throw new Error('插件清单包含空文件路径')
  const resolved = path.resolve(root, value)
  if (!isInside(root, resolved)) throw new Error(`插件清单文件路径越界：${value}`)
  return resolved
}

async function readManifest(pluginRoot) {
  const payload = JSON.parse(await fs.promises.readFile(path.join(pluginRoot, 'manifest.json'), 'utf8'))
  if (
    payload.schema_version !== 1
    || payload.plugin_id !== PLUGIN_ID
    || payload.version !== PLUGIN_VERSION
    || !Array.isArray(payload.files)
    || payload.files.length === 0
  ) {
    throw new Error('Lightroom 插件清单版本或标识无效')
  }
  const files = [...new Set(['manifest.json', ...payload.files])]
  for (const relative of files) {
    if (!(await regularFile(safeManifestPath(pluginRoot, relative)))) {
      throw new Error(`插件清单文件缺失：${relative}`)
    }
  }
  return { payload, files }
}

function createLightroomService(options = {}) {
  const appData = path.resolve(options.appData || process.env.APPDATA || path.join(os.homedir(), 'AppData', 'Roaming'))
  const resources = path.resolve(options.resources || process.resourcesPath || path.resolve(__dirname, '..'))
  const projectRoot = path.resolve(options.projectRoot || path.resolve(__dirname, '..'))
  const platform = options.platform || process.platform
  const programFiles = path.resolve(options.programFiles || process.env.ProgramFiles || 'C:/Program Files')
  const spawn = options.spawn || childProcess.spawn
  const execFile = options.execFile || promisify(childProcess.execFile)
  const modulesRoot = path.resolve(appData, 'Adobe', 'Lightroom', 'Modules')
  const destination = path.resolve(modulesRoot, PLUGIN_DIRECTORY)
  const bridgeRoot = path.resolve(appData, 'Adobe', 'Lightroom', 'YidianPhotoCull', 'lightroom-bridge')

  if (!isInside(modulesRoot, destination)) throw new Error('Lightroom 插件目标路径非法')

  async function validateExecutable(candidate) {
    if (!candidate) return null
    const resolved = path.resolve(String(candidate).replace(/^"|"$/g, '').trim())
    if (path.basename(resolved).toLowerCase() !== 'lightroom.exe') return null
    return (await regularFile(resolved)) ? resolved : null
  }

  async function detectLightroom() {
    if (platform !== 'win32') return { found: false, path: null, source: null }
    const installed = await validateExecutable(path.join(programFiles, 'Adobe', 'Adobe Lightroom Classic', 'Lightroom.exe'))
    if (installed) return { found: true, path: installed, source: 'program-files' }
    try {
      const result = await execFile('reg.exe', ['query', APP_PATHS_KEY, '/ve'], {
        windowsHide: true,
        shell: false,
        encoding: 'utf8',
      })
      const line = String(result.stdout || '').split(/\r?\n/).find((value) => /Lightroom\.exe\s*$/i.test(value))
      const match = line && line.match(/([A-Za-z]:[\\/].*Lightroom\.exe)\s*$/i)
      const registered = await validateExecutable(match && match[1])
      if (registered) return { found: true, path: registered, source: 'app-paths' }
    } catch {
      // 未安装或注册表键不存在时返回明确的未发现状态。
    }
    return { found: false, path: null, source: null }
  }

  async function sourcePluginRoot() {
    const candidates = [
      path.resolve(resources, 'lightroom', PLUGIN_DIRECTORY),
      path.resolve(projectRoot, 'lightroom', PLUGIN_DIRECTORY),
    ]
    for (const candidate of candidates) {
      if (await regularFile(path.join(candidate, 'manifest.json'))) return candidate
    }
    throw new Error('安装资源中缺少 Lightroom 插件')
  }

  async function pluginStatus() {
    try {
      const manifest = JSON.parse(await fs.promises.readFile(path.join(destination, 'manifest.json'), 'utf8'))
      return {
        installed: manifest.plugin_id === PLUGIN_ID,
        compatible: manifest.plugin_id === PLUGIN_ID && manifest.version === PLUGIN_VERSION,
        version: manifest.version || null,
        path: destination,
      }
    } catch {
      return { installed: false, compatible: false, version: null, path: destination }
    }
  }

  async function installPlugin() {
    const source = await sourcePluginRoot()
    const { payload, files } = await readManifest(source)
    await fs.promises.mkdir(modulesRoot, { recursive: true })
    const staging = path.resolve(modulesRoot, `.${PLUGIN_DIRECTORY}.stage-${crypto.randomUUID()}`)
    if (!isInside(modulesRoot, staging)) throw new Error('Lightroom 插件暂存路径非法')
    let backup = null
    try {
      await fs.promises.mkdir(staging, { recursive: false })
      for (const relative of files) {
        const sourceFile = safeManifestPath(source, relative)
        const targetFile = safeManifestPath(staging, relative)
        await fs.promises.mkdir(path.dirname(targetFile), { recursive: true })
        await fs.promises.copyFile(sourceFile, targetFile, fs.constants.COPYFILE_EXCL)
      }
      await readManifest(staging)
      if (fs.existsSync(destination)) {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
        backup = path.resolve(modulesRoot, `${PLUGIN_DIRECTORY}.backup-${timestamp}-${crypto.randomUUID()}`)
        if (!isInside(modulesRoot, backup)) throw new Error('Lightroom 插件备份路径非法')
        await fs.promises.rename(destination, backup)
      }
      await fs.promises.rename(staging, destination)
      return { installed: true, version: payload.version, path: destination, backup }
    } catch (error) {
      if (!fs.existsSync(destination) && backup && fs.existsSync(backup)) {
        await fs.promises.rename(backup, destination).catch(() => undefined)
      }
      if (fs.existsSync(staging)) await fs.promises.rm(staging, { recursive: true, force: true })
      throw error
    }
  }

  async function launchLightroom() {
    const detected = await detectLightroom()
    if (!detected.found || !detected.path) throw new Error('未找到 Adobe Lightroom Classic')
    const child = spawn(detected.path, [], {
      detached: true,
      windowsHide: true,
      shell: false,
      stdio: 'ignore',
    })
    if (typeof child.unref === 'function') child.unref()
    return detected
  }

  async function logsPath() {
    const target = path.join(bridgeRoot, 'outbox')
    await fs.promises.mkdir(target, { recursive: true })
    return target
  }

  return {
    detectLightroom,
    pluginStatus,
    installPlugin,
    launchLightroom,
    logsPath,
    paths: { appData, modulesRoot, destination, bridgeRoot },
  }
}

module.exports = {
  APP_PATHS_KEY,
  PLUGIN_ID,
  PLUGIN_VERSION,
  createLightroomService,
}
