import { FolderOpen } from 'lucide-react'
import type { ProjectSummary } from '../types'
import { Button } from '../components/ui/button'

interface HomePageProps {
  projects: ProjectSummary[]
  onStart: () => void
  onOpenProject: (id: string) => void
}

function formatCreatedAt(value: string): string {
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value || '—'
  return new Intl.DateTimeFormat('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  }).format(parsed)
}

export default function HomePage({ projects, onStart, onOpenProject }: HomePageProps) {
  return (
    <main className="page page--projects">
      <header className="page-toolbar">
        <h1>项目</h1>
        <Button type="button" onClick={onStart}>
          <FolderOpen size={17} />
          选择照片目录
        </Button>
      </header>

      <div className="project-table-wrap">
        <table className="project-table">
          <thead>
            <tr>
              <th scope="col">项目</th>
              <th scope="col">路径</th>
              <th scope="col">照片</th>
              <th scope="col">3星</th>
              <th scope="col">扫描时间</th>
              <th scope="col">状态</th>
            </tr>
          </thead>
          <tbody>
            {projects.length === 0 ? (
              <tr>
                <td className="project-table__empty" colSpan={6}>暂无项目</td>
              </tr>
            ) : projects.map((project) => (
              <tr key={project.id}>
                <td>
                  <button type="button" className="project-link" onClick={() => onOpenProject(project.id)}>
                    {project.name}
                  </button>
                </td>
                <td className="project-path" title={project.source_name}>{project.source_name || '—'}</td>
                <td>{project.total.toLocaleString()}</td>
                <td>{(project.stars_3 ?? project.selected).toLocaleString()}</td>
                <td><time dateTime={project.created_at}>{formatCreatedAt(project.created_at)}</time></td>
                <td><span className="project-status">完成</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </main>
  )
}
