import { useCallback, useEffect, useMemo, useState } from 'react'
import type { PhotoResult, PhotoStars, ScanResults } from '../../types'

export type ResultsWorkspaceMode = 'grid' | 'loupe' | 'compare'

const ALL_STARS: PhotoStars[] = [0, 1, 2, 3]

function firstVisibleId(photos: PhotoResult[]): string | null {
  return photos[0]?.id ?? null
}

export function useResultsWorkspace(results: ScanResults) {
  const [starFilters, setStarFilters] = useState<Set<PhotoStars>>(() => new Set())
  const [personFilter, setPersonFilter] = useState('all')
  const [stageFilter, setStageFilter] = useState('all')
  const [selectedIds, setSelectedIds] = useState<Set<string>>(() => new Set())
  const [activeId, setActiveId] = useState<string | null>(() => firstVisibleId(results.photos))
  const [mode, setMode] = useState<ResultsWorkspaceMode>('grid')

  const people = useMemo(
    () => Array.from(new Set(results.photos.flatMap((photo) => photo.person_ids))).sort(),
    [results.photos],
  )

  const stages = useMemo(() => {
    const labels = new Map<string, string>()
    for (const photo of results.photos) {
      if (photo.stage_id) labels.set(photo.stage_id, photo.stage_label || photo.stage_id)
    }
    return Array.from(labels, ([id, label]) => ({ id, label }))
  }, [results.photos])

  const visiblePhotos = useMemo(() => results.photos.filter((photo) => {
    const starMatches = starFilters.size === 0 || starFilters.has(photo.stars)
    const personMatches = personFilter === 'all' || photo.person_ids.includes(personFilter)
    const stageMatches = stageFilter === 'all' || photo.stage_id === stageFilter
    return starMatches && personMatches && stageMatches
  }), [personFilter, results.photos, stageFilter, starFilters])

  useEffect(() => {
    setStarFilters(new Set())
    setPersonFilter('all')
    setStageFilter('all')
    setSelectedIds(new Set())
    setActiveId(firstVisibleId(results.photos))
    setMode('grid')
  }, [results.project_id])

  useEffect(() => {
    if (!visiblePhotos.some((photo) => photo.id === activeId)) {
      setActiveId(firstVisibleId(visiblePhotos))
    }
  }, [activeId, visiblePhotos])

  const changeStarFilters = useCallback((stars: PhotoStars[]) => {
    const exactStars = stars.filter((star) => ALL_STARS.includes(star))
    setStarFilters(new Set(exactStars))
  }, [])

  const toggleSelected = useCallback((photoId: string) => {
    setSelectedIds((current) => {
      const next = new Set(current)
      if (next.has(photoId)) next.delete(photoId)
      else next.add(photoId)
      return next
    })
  }, [])

  const move = useCallback((delta: number) => {
    if (!visiblePhotos.length) return
    const index = Math.max(0, visiblePhotos.findIndex((photo) => photo.id === activeId))
    const nextIndex = (index + delta + visiblePhotos.length) % visiblePhotos.length
    setActiveId(visiblePhotos[nextIndex].id)
  }, [activeId, visiblePhotos])

  const activePhoto = useMemo(
    () => visiblePhotos.find((photo) => photo.id === activeId) ?? visiblePhotos[0] ?? null,
    [activeId, visiblePhotos],
  )

  const comparePhoto = useMemo(() => {
    if (!activePhoto) return null
    const selected = visiblePhotos.filter((photo) => selectedIds.has(photo.id) && photo.id !== activePhoto.id)
    if (selected.length) return selected[0]
    const activeIndex = visiblePhotos.findIndex((photo) => photo.id === activePhoto.id)
    return visiblePhotos.length > 1 ? visiblePhotos[(activeIndex + 1) % visiblePhotos.length] : null
  }, [activePhoto, selectedIds, visiblePhotos])

  return {
    starFilters,
    changeStarFilters,
    personFilter,
    setPersonFilter,
    stageFilter,
    setStageFilter,
    people,
    stages,
    selectedIds,
    toggleSelected,
    activeId,
    setActiveId,
    activePhoto,
    comparePhoto,
    mode,
    setMode,
    move,
    visiblePhotos,
  }
}
