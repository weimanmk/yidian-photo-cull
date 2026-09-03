import { AlertCircle, FolderOpen, Pause, Play, ScanLine } from 'lucide-react'
import type { EngineSettings, ScanStatus } from '../types'
import { Button } from '../components/ui/button'
import { Progress } from '../components/ui/progress'
import { ToggleGroup, ToggleGroupItem } from '../components/ui/toggle-group'

interface CullPageProps {
  folder: string
  settings: EngineSettings
  status: ScanStatus
  error: string | null
  provider?: string
  onChooseFolder: () => void
  onChangeSettings: (next: Partial<EngineSettings>) => void
  onStart: () => void
  onCancel: () => void
  onShowResults: () => void
}

const presets = [
  { key: 'cautious' as const, label: '谨慎' },
  { key: 'balanced' as const, label: '均衡' },
  { key: 'aggressive' as const, label: '积极' },
]

function formatElapsed(seconds: number): string {
  const normalized = Math.max(0, Math.floor(seconds))
  const minutes = Math.floor(normalized / 60)
  const remainder = normalized % 60
  return `${minutes}:${String(remainder).padStart(2, '0')}`
}

export default function CullPage({
  folder,
  settings,
  status,
  error,
  provider = '自动',
  onChooseFolder,
  onChangeSettings,
  onStart,
  onCancel,
  onShowResults,
}: CullPageProps) {
  const busy = !['idle', 'completed', 'cancelled', 'failed'].includes(status.status)
  const done = status.status === 'completed'

  return (
    <main className="page page--cull">
      <header className="cull-toolbar">
        <h1>筛图</h1>
        <Button type="button" variant="outline" disabled={busy} onClick={onChooseFolder}>
          <FolderOpen size={17} />
          <span className="cull-toolbar__path">{folder || '选择照片目录'}</span>
        </Button>
        <ToggleGroup
          type="single"
          aria-label="相似度策略"
          value={settings.grouping_preset}
          onValueChange={(value) => {
            if (value) onChangeSettings({ grouping_preset: value as EngineSettings['grouping_preset'] })
          }}
          disabled={busy}
        >
          {presets.map((preset) => (
            <ToggleGroupItem key={preset.key} value={preset.key} aria-label={preset.label}>
              {preset.label}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </header>

      <section className="cull-options" aria-label="扫描选项">
        <label>
          <input
            type="checkbox"
            checked={settings.recursive}
            disabled={busy}
            onChange={(event) => onChangeSettings({ recursive: event.target.checked })}
          />
          包含子目录
        </label>
        <label>
          环节跨度
          <select
            value={settings.coverage_window_minutes}
            disabled={busy}
            onChange={(event) => onChangeSettings({ coverage_window_minutes: Number(event.target.value) })}
          >
            {[5, 10, 15, 20, 30, 60].map((value) => <option key={value} value={value}>{value} 分钟</option>)}
          </select>
        </label>
        <span className="coverage-status">人物×环节：开启</span>
      </section>

      <section className="scan-panel" aria-label="筛图状态">
        <dl className="scan-facts">
          <div><dt>阶段</dt><dd>{done ? '完成' : status.phase || '等待'}</dd></div>
          <div><dt>进度</dt><dd>{status.processed.toLocaleString()} / {status.total.toLocaleString()}</dd></div>
          <div><dt>耗时</dt><dd>{formatElapsed(status.elapsed_seconds)}</dd></div>
          <div><dt>GPU</dt><dd>{provider}</dd></div>
        </dl>

        <Progress value={status.progress} aria-label="筛图进度" />

        <div className="scan-actions">
          {done ? (
            <Button type="button" onClick={onShowResults}><Play size={17} />查看结果</Button>
          ) : busy ? (
            <Button type="button" variant="outline" onClick={onCancel}><Pause size={17} />取消</Button>
          ) : (
            <Button type="button" disabled={!folder} onClick={onStart}><ScanLine size={17} />开始筛图</Button>
          )}
        </div>

        {error ? <div className="error-row" role="alert"><AlertCircle size={17} />{error}</div> : null}

        <details className="runtime-log">
          <summary>运行日志</summary>
          <pre>{[status.message, status.current_file, error].filter(Boolean).join('\n') || '等待开始'}</pre>
        </details>
      </section>
    </main>
  )
}
