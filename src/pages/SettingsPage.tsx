import { RotateCcw, Wrench } from 'lucide-react'
import type { Appearance, ThemeMode, UiScale } from '../appearance'
import { Button } from '../components/ui/button'
import type { EngineSettings, HealthResponse } from '../types'

export interface LightroomSettingsStatus {
  executable: string | null
  pluginVersion: string | null
  pluginCompatible: boolean
  heartbeatAt: string | null
  heartbeatReady: boolean
}

interface SettingsPageProps {
  settings: EngineSettings
  health: HealthResponse | null
  appearance: Appearance
  lightroom: LightroomSettingsStatus | null
  repairingLightroom: boolean
  onAppearanceChange: (appearance: Appearance) => void
  onChange: (next: Partial<EngineSettings>) => void
  onSave: () => void
  onRestart: () => void
  onRepairLightroom: () => void
}

const themes: Array<{ value: ThemeMode; label: string }> = [
  { value: 'light', label: '白天' },
  { value: 'dark', label: '夜间' },
  { value: 'system', label: '跟随系统' },
]

const scales: Array<{ value: UiScale; label: string }> = [
  { value: 1, label: '100%' },
  { value: 1.1, label: '110%' },
  { value: 1.25, label: '125%' },
]

function StatusValue({ ready, children }: { ready: boolean; children: React.ReactNode }) {
  return <span className={ready ? 'settings-status is-ready' : 'settings-status'}>{children}</span>
}

export default function SettingsPage({
  settings,
  health,
  appearance,
  lightroom,
  repairingLightroom,
  onAppearanceChange,
  onChange,
  onSave,
  onRestart,
  onRepairLightroom,
}: SettingsPageProps) {
  const provider = settings.use_gpu
    ? ([health?.face_ai, health?.body_ai, health?.depth_ai].find((model) => model?.provider_source === 'actual')?.backend ?? health?.face_ai.backend ?? '检测中')
    : 'CPUExecutionProvider'

  return (
    <main className="page page--settings">
      <header className="settings-toolbar">
        <h1>设置</h1>
        <Button onClick={onSave}>保存设置</Button>
      </header>

      <div className="settings-sections">
        <section className="settings-section">
          <h2>外观</h2>
          <div className="settings-row">
            <span>主题</span>
            <div className="settings-segment" role="group" aria-label="界面主题">
              {themes.map((theme) => (
                <Button
                  key={theme.value}
                  size="sm"
                  variant={appearance.theme === theme.value ? 'default' : 'outline'}
                  aria-pressed={appearance.theme === theme.value}
                  onClick={() => onAppearanceChange({ ...appearance, theme: theme.value })}
                >
                  {theme.label}
                </Button>
              ))}
            </div>
          </div>
          <div className="settings-row">
            <span>界面比例</span>
            <div className="settings-segment" role="group" aria-label="界面比例">
              {scales.map((scale) => (
                <Button
                  key={scale.value}
                  size="sm"
                  variant={appearance.scale === scale.value ? 'default' : 'outline'}
                  aria-pressed={appearance.scale === scale.value}
                  onClick={() => onAppearanceChange({ ...appearance, scale: scale.value })}
                >
                  {scale.label}
                </Button>
              ))}
            </div>
          </div>
        </section>

        <section className="settings-section">
          <h2>性能</h2>
          <label className="settings-row settings-row--switch">
            <span>CUDA</span>
            <b>{provider}</b>
            <input
              type="checkbox"
              aria-label="优先使用 CUDA"
              checked={settings.use_gpu}
              onChange={(event) => onChange({ use_gpu: event.target.checked })}
            />
          </label>
          <label className="settings-row settings-row--range">
            <span>预览质量</span>
            <input
              type="range"
              aria-label="预览质量"
              min="60"
              max="95"
              step="5"
              value={settings.jpeg_preview_quality}
              onChange={(event) => onChange({ jpeg_preview_quality: Number(event.target.value) })}
            />
            <b>{settings.jpeg_preview_quality}</b>
          </label>
          <div className="settings-row">
            <span>AI 引擎</span>
            <Button size="sm" variant="outline" onClick={onRestart}><RotateCcw size={15} />重启引擎</Button>
          </div>
        </section>

        <section className="settings-section">
          <h2>模型</h2>
          <div className="settings-row"><span>人脸与人物</span><StatusValue ready={Boolean(health?.face_ai.available)}>{health?.face_ai.backend ?? '不可用'}</StatusValue></div>
          <div className="settings-row"><span>人体与动作</span><StatusValue ready={Boolean(health?.body_ai?.available && health.pose_ai?.available)}>{health?.pose_ai?.model ?? '不可用'}</StatusValue></div>
          <div className="settings-row"><span>3D 关键点</span><StatusValue ready={Boolean(health?.face_ai.landmark_3d_model?.available)}>{health?.face_ai.landmark_3d_model?.name ?? '不可用'}</StatusValue></div>
          <div className="settings-row"><span>景深</span><StatusValue ready={Boolean(health?.depth_ai?.available)}>{health?.depth_ai?.model ?? '不可用'}</StatusValue></div>
          <div className="settings-row"><span>场景语义</span><StatusValue ready={Boolean(health?.scene_ai?.available)}>{health?.scene_ai?.backend ?? '不可用'}</StatusValue></div>
          <div className="settings-row"><span>视觉大模型</span><StatusValue ready={Boolean(health?.vlm_ai?.available)}>{health?.vlm_ai?.model_id ?? '未启用'}</StatusValue></div>
          <label className="settings-row settings-row--range">
            <span>人物阈值</span>
            <input
              type="range"
              aria-label="人物身份阈值"
              min="0.32"
              max="0.62"
              step="0.01"
              value={settings.face_identity_threshold}
              onChange={(event) => onChange({ face_identity_threshold: Number(event.target.value) })}
            />
            <b>{settings.face_identity_threshold.toFixed(2)}</b>
          </label>
        </section>

        <section className="settings-section">
          <h2>Lightroom</h2>
          <div className="settings-row"><span>可执行文件</span><b title={lightroom?.executable ?? ''}>{lightroom?.executable ?? '未找到'}</b></div>
          <div className="settings-row"><span>插件版本</span><StatusValue ready={Boolean(lightroom?.pluginCompatible)}>{lightroom?.pluginVersion ?? '未安装'}</StatusValue></div>
          <div className="settings-row"><span>心跳</span><StatusValue ready={Boolean(lightroom?.heartbeatReady)}>{lightroom?.heartbeatReady ? '心跳正常' : '等待 Lightroom'}</StatusValue></div>
          <div className="settings-row">
            <span>插件</span>
            <Button size="sm" variant="outline" disabled={repairingLightroom} onClick={onRepairLightroom}><Wrench size={15} />修复插件</Button>
          </div>
        </section>
      </div>
    </main>
  )
}
