import { ChevronLeft, ChevronRight } from 'lucide-react'
import { useCallback, useState } from 'react'
import { assetUrl } from '../api'
import { Button } from '../components/ui/button'
import PhotoGrid from '../features/results/photo-grid'
import PhotoInspector from '../features/results/photo-inspector'
import ResultsToolbar from '../features/results/results-toolbar'
import useCullingShortcuts from '../features/results/use-culling-shortcuts'
import { useResultsWorkspace } from '../features/results/use-results-workspace'
import type { PhotoResult, PhotoStars, ScanResults } from '../types'

interface ResultsPageProps {
  results: ScanResults | null
  onRate: (photoId: string, stars: PhotoStars, locked: true) => void | Promise<void>
  onImportLightroom: () => void
  onExportFolder: () => void
}

function PhotoPreview({ photo, label }: { photo: PhotoResult; label?: string }) {
  return (
    <figure className="workspace-preview">
      {label && <figcaption>{label}</figcaption>}
      <img src={assetUrl(photo.image_url)} alt={photo.filename} />
      <footer>
        <span>{photo.filename}</span>
        <span className="workspace-preview__stars">{photo.stars === 0 ? '0星' : `${'★'.repeat(photo.stars)} ${photo.stars}星`}</span>
      </footer>
    </figure>
  )
}

export default function ResultsPage({ results, onRate, onImportLightroom, onExportFolder }: ResultsPageProps) {
  if (!results) {
    return <main className="page page--empty-results"><h1>暂无结果</h1></main>
  }

  return (
    <ResultsWorkspace
      results={results}
      onRate={onRate}
      onImportLightroom={onImportLightroom}
      onExportFolder={onExportFolder}
    />
  )
}

function ResultsWorkspace({ results, onRate, onImportLightroom, onExportFolder }: ResultsPageProps & { results: ScanResults }) {
  const workspace = useResultsWorkspace(results)
  const [inspectorCollapsed, setInspectorCollapsed] = useState(false)
  const rateActive = useCallback((stars: PhotoStars) => {
    if (workspace.activePhoto) void onRate(workspace.activePhoto.id, stars, true)
  }, [onRate, workspace.activePhoto])
  const moveNext = useCallback(() => workspace.move(1), [workspace.move])
  const movePrevious = useCallback(() => workspace.move(-1), [workspace.move])

  useCullingShortcuts({
    enabled: Boolean(workspace.activePhoto),
    onRate: rateActive,
    onNext: moveNext,
    onPrevious: movePrevious,
    onModeChange: workspace.setMode,
  })

  return (
    <main className="results-workspace">
      <ResultsToolbar
        projectName={results.project_name}
        summary={results.summary}
        starFilters={workspace.starFilters}
        onStarFiltersChange={workspace.changeStarFilters}
        people={workspace.people}
        personFilter={workspace.personFilter}
        onPersonFilterChange={workspace.setPersonFilter}
        stages={workspace.stages}
        stageFilter={workspace.stageFilter}
        onStageFilterChange={workspace.setStageFilter}
        mode={workspace.mode}
        onModeChange={workspace.setMode}
        onImportLightroom={onImportLightroom}
        onExportFolder={onExportFolder}
      />

      <div className="results-body">
        <section className={`results-surface results-surface--${workspace.mode}`}>
          {workspace.mode === 'grid' && (
            <PhotoGrid
              photos={workspace.visiblePhotos}
              activeId={workspace.activeId}
              selectedIds={workspace.selectedIds}
              onActivate={workspace.setActiveId}
              onToggleSelected={workspace.toggleSelected}
            />
          )}

          {workspace.mode === 'loupe' && workspace.activePhoto && (
            <div className="loupe-view">
              <PhotoPreview photo={workspace.activePhoto} />
              <Button variant="ghost" size="icon" aria-label="上一张" onClick={movePrevious}><ChevronLeft /></Button>
              <Button variant="ghost" size="icon" aria-label="下一张" onClick={moveNext}><ChevronRight /></Button>
            </div>
          )}

          {workspace.mode === 'compare' && workspace.activePhoto && (
            <div className="compare-view">
              <PhotoPreview photo={workspace.activePhoto} label="当前" />
              {workspace.comparePhoto && <PhotoPreview photo={workspace.comparePhoto} label="候选" />}
            </div>
          )}

          {!workspace.activePhoto && <div className="results-empty">当前筛选无照片</div>}
        </section>

        <PhotoInspector
          photo={workspace.activePhoto}
          collapsed={inspectorCollapsed}
          onCollapsedChange={setInspectorCollapsed}
          onRate={onRate}
        />
      </div>

      <footer className="results-statusbar">
        <span>{workspace.visiblePhotos.length} / {results.summary.total}</span>
        <span>{workspace.selectedIds.size} 已选择</span>
      </footer>
    </main>
  )
}
