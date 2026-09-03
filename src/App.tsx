import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { CloudOff, LoaderCircle } from 'lucide-react'
import { api, initializeApi } from './api'
import Sidebar from './components/Sidebar'
import TitleBar from './components/TitleBar'
import CullPage from './pages/CullPage'
import HomePage from './pages/HomePage'
import ResultsPage from './pages/ResultsPage'
import SettingsPage, { type LightroomSettingsStatus } from './pages/SettingsPage'
import FolderExportDialog from './features/export/folder-export-dialog'
import LightroomImportDialog from './features/lightroom/lightroom-import-dialog'
import { loadAppearance, saveAppearance, stepUiScale, type Appearance } from './appearance'
import { applySemanticRating } from './features/results/apply-semantic-rating'
import type { EngineSettings, HealthResponse, PhotoStars, ProjectSummary, ScanResults, ScanStatus, ViewKey } from './types'

const idleStatus: ScanStatus = {
  status: 'idle', phase: '等待', message: '等待开始', processed: 0, total: 0, progress: 0,
  current_file: '', elapsed_seconds: 0, eta_seconds: null, error: null, project_id: null,
}

const defaultSettings: EngineSettings = {
  grouping_preset: 'balanced', keep_per_group: 1,
  coverage_enabled: true, coverage_window_minutes: 15,
  face_identity_threshold: 0.42,
  use_gpu: true, recursive: true, jpeg_preview_quality: 80,
  vlm_enabled: false, vlm_server_url: 'http://127.0.0.1:8768',
  vlm_executable_path: '', vlm_model_path: '', vlm_mmproj_path: '',
  vlm_model_id: 'disabled', vlm_quantization: 'none',
  vlm_context_size: 8192, vlm_gpu_layers: 16, vlm_max_groups: 12,
  vlm_max_candidates: 4, vlm_ambiguity_margin: 8, vlm_min_confidence: 0.65,
  vlm_timeout_seconds: 300,
}

export default function App() {
  const [view, setView] = useState<ViewKey>('home')
  const [appearance, setAppearance] = useState<Appearance>(() => loadAppearance())
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [status, setStatus] = useState<ScanStatus>(idleStatus)
  const [results, setResults] = useState<ScanResults | null>(null)
  const [projects, setProjects] = useState<ProjectSummary[]>([])
  const [settings, setSettings] = useState<EngineSettings>(defaultSettings)
  const [folder, setFolder] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [booting, setBooting] = useState(true)
  const [lightroomDialogOpen, setLightroomDialogOpen] = useState(false)
  const [folderExportDialogOpen, setFolderExportDialogOpen] = useState(false)
  const [lightroomSettings, setLightroomSettings] = useState<LightroomSettingsStatus | null>(null)
  const [repairingLightroom, setRepairingLightroom] = useState(false)
  const polling = useRef<number | null>(null)

  const busy = useMemo(() => !['idle', 'completed', 'cancelled', 'failed'].includes(status.status), [status.status])

  useEffect(() => {
    saveAppearance(appearance)
  }, [appearance])

  useEffect(() => {
    const handleZoom = (event: KeyboardEvent) => {
      if (!event.ctrlKey || event.metaKey || event.altKey) return
      if (event.key === '0') {
        event.preventDefault()
        setAppearance((current) => ({ ...current, scale: 1 }))
        return
      }
      if (event.key === '+' || event.key === '=') {
        event.preventDefault()
        setAppearance((current) => ({ ...current, scale: stepUiScale(current.scale, 1) }))
        return
      }
      if (event.key === '-' || event.key === '_') {
        event.preventDefault()
        setAppearance((current) => ({ ...current, scale: stepUiScale(current.scale, -1) }))
      }
    }
    window.addEventListener('keydown', handleZoom)
    return () => window.removeEventListener('keydown', handleZoom)
  }, [])

  const refreshProjects = useCallback(async () => {
    const next = await api.projects()
    setProjects(next)
  }, [])

  const refreshLightroomSettings = useCallback(async () => {
    if (!window.desktop) {
      setLightroomSettings(null)
      return
    }
    const [desktopStatus, backendStatus] = await Promise.all([
      window.desktop.lightroomStatus(),
      api.lightroomStatus(),
    ])
    const heartbeat = backendStatus.plugin_heartbeat
    const heartbeatTime = heartbeat ? Date.parse(heartbeat.timestamp) : Number.NaN
    setLightroomSettings({
      executable: desktopStatus.lightroom.path,
      pluginVersion: desktopStatus.plugin.version,
      pluginCompatible: desktopStatus.plugin.compatible,
      heartbeatAt: heartbeat?.timestamp ?? null,
      heartbeatReady: Boolean(
        heartbeat?.running
        && heartbeat.plugin_id === 'com.yidian.photocull.lightroom'
        && Number.isFinite(heartbeatTime)
        && Math.abs(Date.now() - heartbeatTime) <= 15_000,
      ),
    })
  }, [])

  useEffect(() => {
    if (view === 'settings') void refreshLightroomSettings().catch(() => setLightroomSettings(null))
  }, [refreshLightroomSettings, view])

  const loadResults = useCallback(async () => {
    const next = await api.results()
    setResults(next)
    await refreshProjects()
    return next
  }, [refreshProjects])

  useEffect(() => {
    let cancelled = false
    async function boot() {
      await initializeApi()
      for (let attempt = 0; attempt < 20 && !cancelled; attempt += 1) {
        try {
          const [healthData, settingsData, statusData, projectData] = await Promise.all([api.health(), api.settings(), api.status(), api.projects()])
          if (cancelled) return
          setHealth(healthData)
          setSettings(settingsData)
          setStatus(statusData)
          setProjects(projectData)
          if (statusData.status === 'completed') await loadResults().catch(() => undefined)
          setBooting(false)
          return
        } catch {
          await new Promise((resolve) => setTimeout(resolve, 500))
        }
      }
      if (!cancelled) {
        setError('本地 AI 服务未启动。请检查 Python 环境后重启引擎。')
        setBooting(false)
      }
    }
    boot()
    return () => { cancelled = true }
  }, [loadResults])

  useEffect(() => {
    if (!busy) {
      if (polling.current) window.clearInterval(polling.current)
      polling.current = null
      return
    }
    polling.current = window.setInterval(async () => {
      try {
        const next = await api.status()
        setStatus(next)
        if (next.status === 'completed') {
          await loadResults()
          setView('results')
        }
        if (next.status === 'failed') setError(next.error || '筛选任务失败')
      } catch (requestError) {
        setError(requestError instanceof Error ? requestError.message : '无法获取任务状态')
      }
    }, 650)
    return () => {
      if (polling.current) window.clearInterval(polling.current)
      polling.current = null
    }
  }, [busy, loadResults])

  async function chooseFolder() {
    if (!window.desktop) {
      setError('浏览器预览模式不能读取本机路径，请在 Electron 桌面端操作。')
      return
    }
    const selected = await window.desktop.selectFolder()
    if (selected) {
      setFolder(selected)
      setError(null)
    }
  }

  async function startScan() {
    if (!folder) return
    setError(null)
    try {
      const next = await api.startScan(folder, { ...settings, coverage_enabled: true })
      setStatus(next)
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : '无法启动筛选')
    }
  }

  async function openProject(projectId: string) {
    try {
      const project = await api.loadProject(projectId)
      setResults(project)
      setView('results')
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : '项目读取失败')
    }
  }

  async function ratePhoto(photoId: string, stars: PhotoStars, locked: true) {
    const snapshot = results
    if (!snapshot) return

    setResults(applySemanticRating(snapshot, photoId, stars, locked))
    try {
      await api.ratePhoto(photoId, stars, locked)
    } catch (requestError) {
      setResults((current) => current?.project_id === snapshot.project_id ? snapshot : current)
      setError(requestError instanceof Error ? requestError.message : '人工星级保存失败')
      return
    }

    void api.results()
      .then((saved) => setResults((current) => current?.project_id === saved.project_id ? saved : current))
      .then(refreshProjects)
      .catch(() => setError('星级已保存，但项目汇总刷新失败。'))
  }

  async function saveSettings() {
    try {
      const next = await api.updateSettings(settings)
      const nextHealth = await api.health()
      setSettings(next)
      setHealth(nextHealth)
      setError('设置已保存在本机。')
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : '设置保存失败')
    }
  }

  async function repairLightroomPlugin() {
    if (!window.desktop || repairingLightroom) return
    setRepairingLightroom(true)
    try {
      await window.desktop.installLightroomPlugin()
      await refreshLightroomSettings()
      setError('Lightroom 插件已修复。')
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'Lightroom 插件修复失败')
    } finally {
      setRepairingLightroom(false)
    }
  }

  if (booting) {
    return <div className="boot-screen"><div className="brand-mark">点</div><LoaderCircle /><strong>正在连接本地 AI 引擎</strong><span>照片不会离开这台电脑</span></div>
  }

  return (
    <div className={`app-shell ${window.desktop ? 'is-desktop' : ''}`}>
      <TitleBar appearance={appearance} onAppearanceChange={setAppearance} />
      <Sidebar active={view} onNavigate={setView} aiAvailable={Boolean(health?.face_ai.available)} busy={busy} />
      <div className="app-content">
        {error && <button className="global-notice" onClick={() => setError(null)}><CloudOff size={15} /><span>{error}</span><b>×</b></button>}
        {view === 'home' && <HomePage projects={projects} onStart={() => setView('cull')} onOpenProject={openProject} />}
        {view === 'cull' && <CullPage folder={folder} settings={settings} status={status} error={status.error} provider={settings.use_gpu ? ([health?.depth_ai, health?.face_ai, health?.body_ai].find((model) => model?.provider_source === 'actual')?.backend ?? '检测中') : 'CPU'} onChooseFolder={chooseFolder} onChangeSettings={(next) => setSettings((current) => ({ ...current, ...next }))} onStart={startScan} onCancel={async () => setStatus(await api.cancelScan())} onShowResults={() => setView('results')} />}
        {view === 'results' && <ResultsPage results={results} onRate={ratePhoto} onImportLightroom={() => setLightroomDialogOpen(true)} onExportFolder={() => setFolderExportDialogOpen(true)} />}
        {view === 'settings' && <SettingsPage settings={settings} health={health} appearance={appearance} lightroom={lightroomSettings} repairingLightroom={repairingLightroom} onAppearanceChange={setAppearance} onChange={(next) => setSettings((current) => ({ ...current, ...next }))} onSave={saveSettings} onRestart={async () => { await window.desktop?.restartBackend(); window.location.reload() }} onRepairLightroom={repairLightroomPlugin} />}
        {results && (
          <>
            <LightroomImportDialog projectId={results.project_id} desktop={window.desktop ?? null} open={lightroomDialogOpen} onOpenChange={setLightroomDialogOpen} />
            <FolderExportDialog projectId={results.project_id} desktop={window.desktop ?? null} open={folderExportDialogOpen} onOpenChange={setFolderExportDialogOpen} />
          </>
        )}
      </div>
    </div>
  )
}
