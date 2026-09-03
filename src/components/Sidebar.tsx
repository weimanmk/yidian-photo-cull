import { FolderSearch2, Images, LayoutGrid, Settings2 } from 'lucide-react'
import type { ViewKey } from '../types'
import { Button } from './ui/button'

interface SidebarProps {
  active: ViewKey
  onNavigate: (view: ViewKey) => void
  aiAvailable?: boolean
  busy: boolean
}

const items: Array<{ key: ViewKey; label: string; icon: typeof LayoutGrid }> = [
  { key: 'home', label: '项目', icon: LayoutGrid },
  { key: 'cull', label: '筛图', icon: FolderSearch2 },
  { key: 'results', label: '结果', icon: Images },
  { key: 'settings', label: '设置', icon: Settings2 },
]

export default function Sidebar({ active, onNavigate, aiAvailable = false, busy }: SidebarProps) {
  return (
    <aside className="sidebar">
      <div className="sidebar__logo">
        <span className="brand-mark">点</span>
        <strong>一点筛图</strong>
      </div>

      <nav className="sidebar__nav" aria-label="主导航">
        {items.map(({ key, label, icon: Icon }) => (
          <Button
            key={key}
            type="button"
            variant="ghost"
            className={active === key ? 'is-active' : undefined}
            aria-current={active === key ? 'page' : undefined}
            onClick={() => onNavigate(key)}
          >
            <Icon size={18} strokeWidth={1.8} />
            <span>{label}</span>
            {key === 'cull' && busy ? <i className="nav-pulse" aria-hidden="true" /> : null}
          </Button>
        ))}
      </nav>

      <div className="sidebar__status" aria-label={aiAvailable ? '本地 AI 已就绪' : '本地 AI 基础模式'}>
        <i className={aiAvailable ? 'is-ready' : undefined} />
        <span>{aiAvailable ? '就绪' : '基础'}</span>
      </div>
    </aside>
  )
}
