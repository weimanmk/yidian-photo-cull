import { Maximize2, Minus, X } from 'lucide-react'
import type { Appearance } from '../appearance'

interface TitleBarProps {
  appearance: Appearance
  onAppearanceChange: (appearance: Appearance) => void
}

export default function TitleBar({ appearance }: TitleBarProps) {
  if (!window.desktop) return null
  return (
    <header className="titlebar" data-theme-mode={appearance.theme}>
      <div className="titlebar__brand">
        <span className="brand-mark brand-mark--small">点</span>
        <span>一点筛图</span>
      </div>
      <div className="titlebar__drag" />
      <div className="window-actions">
        <button type="button" aria-label="最小化" title="最小化" onClick={() => window.desktop?.minimize()}>
          <Minus size={16} />
        </button>
        <button type="button" aria-label="最大化" title="最大化" onClick={() => window.desktop?.maximize()}>
          <Maximize2 size={14} />
        </button>
        <button type="button" className="window-actions__close" aria-label="关闭" title="关闭" onClick={() => window.desktop?.close()}>
          <X size={17} />
        </button>
      </div>
    </header>
  )
}
