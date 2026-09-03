import { FolderOutput, Import, UserRound } from 'lucide-react'
import { Button } from '../../components/ui/button'
import { ToggleGroup, ToggleGroupItem } from '../../components/ui/toggle-group'
import type { PhotoStars, ScanSummary } from '../../types'
import type { ResultsWorkspaceMode } from './use-results-workspace'

interface ResultsToolbarProps {
  projectName: string
  summary: ScanSummary
  starFilters: Set<PhotoStars>
  onStarFiltersChange: (stars: PhotoStars[]) => void
  people: string[]
  personFilter: string
  onPersonFilterChange: (personId: string) => void
  stages: Array<{ id: string; label: string }>
  stageFilter: string
  onStageFilterChange: (stageId: string) => void
  mode: ResultsWorkspaceMode
  onModeChange: (mode: ResultsWorkspaceMode) => void
  onImportLightroom: () => void
  onExportFolder: () => void
}

const stars: PhotoStars[] = [0, 1, 2, 3]

export default function ResultsToolbar({
  projectName,
  summary,
  starFilters,
  onStarFiltersChange,
  people,
  personFilter,
  onPersonFilterChange,
  stages,
  stageFilter,
  onStageFilterChange,
  mode,
  onModeChange,
  onImportLightroom,
  onExportFolder,
}: ResultsToolbarProps) {
  return (
    <header className="results-toolbar">
      <h1 title={projectName}>{projectName}</h1>

      <ToggleGroup
        type="multiple"
        size="sm"
        aria-label="星级筛选"
        value={Array.from(starFilters, String)}
        onValueChange={(values) => onStarFiltersChange(values.map(Number) as PhotoStars[])}
      >
        {stars.map((star) => (
          <ToggleGroupItem key={star} value={String(star)} aria-label={`${star}星`}>
            <span className="results-toolbar__star">{star}星</span>
            <small>{summary[`stars_${star}`]}</small>
          </ToggleGroupItem>
        ))}
      </ToggleGroup>

      <label className="results-filter-select">
        <UserRound size={15} aria-hidden="true" />
        <select aria-label="人物" value={personFilter} onChange={(event) => onPersonFilterChange(event.target.value)}>
          <option value="all">全部人物</option>
          {people.map((personId) => <option key={personId} value={personId}>{personId}</option>)}
        </select>
      </label>

      <label className="results-filter-select">
        <select aria-label="环节" value={stageFilter} onChange={(event) => onStageFilterChange(event.target.value)}>
          <option value="all">全部环节</option>
          {stages.map((stage) => <option key={stage.id} value={stage.id}>{stage.label}</option>)}
        </select>
      </label>

      <ToggleGroup
        type="single"
        size="sm"
        aria-label="查看模式"
        value={mode}
        onValueChange={(value) => value && onModeChange(value as ResultsWorkspaceMode)}
      >
        <ToggleGroupItem value="grid" aria-label="网格 G">G</ToggleGroupItem>
        <ToggleGroupItem value="loupe" aria-label="单张 E">E</ToggleGroupItem>
        <ToggleGroupItem value="compare" aria-label="对比 C">C</ToggleGroupItem>
      </ToggleGroup>

      <div className="results-toolbar__actions">
        <Button
          size="sm"
          variant="outline"
          aria-label="导入 Lightroom"
          title="导入 Lightroom"
          onClick={onImportLightroom}
        >
          <Import size={15} />
          <span className="results-toolbar__action-label">导入 Lightroom</span>
        </Button>
        <Button size="sm" aria-label="导出文件夹" title="导出文件夹" onClick={onExportFolder}>
          <FolderOutput size={15} />
          <span className="results-toolbar__action-label">导出文件夹</span>
        </Button>
      </div>
    </header>
  )
}
