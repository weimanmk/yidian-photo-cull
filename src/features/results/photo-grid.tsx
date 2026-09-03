import { observeElementRect, useVirtualizer } from '@tanstack/react-virtual'
import { CircleAlert, Check } from 'lucide-react'
import { useLayoutEffect, useMemo, useRef, useState } from 'react'
import { assetUrl } from '../../api'
import type { PhotoResult } from '../../types'

export interface PhotoGridProps {
  photos: PhotoResult[]
  activeId: string | null
  selectedIds: Set<string>
  onActivate: (photoId: string) => void
  onToggleSelected: (photoId: string) => void
}

const DEFAULT_WIDTH = 1_000
const DEFAULT_HEIGHT = 720
const MIN_CELL_WIDTH = 180
const CELL_GAP = 8
const ROW_HEIGHT = 174

export default function PhotoGrid({ photos, activeId, selectedIds, onActivate, onToggleSelected }: PhotoGridProps) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(DEFAULT_WIDTH)

  useLayoutEffect(() => {
    const element = scrollRef.current
    if (!element) return

    const updateWidth = () => setWidth(element.clientWidth || DEFAULT_WIDTH)
    updateWidth()
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(updateWidth)
    observer?.observe(element)
    return () => observer?.disconnect()
  }, [])

  const columns = Math.max(1, Math.floor((width + CELL_GAP) / (MIN_CELL_WIDTH + CELL_GAP)))
  const rowCount = Math.ceil(photos.length / columns)
  const virtualizer = useVirtualizer({
    count: rowCount,
    getScrollElement: () => scrollRef.current,
    observeElementRect: (instance, callback) => observeElementRect(instance, (rect) => callback({
      width: rect.width || DEFAULT_WIDTH,
      height: rect.height || DEFAULT_HEIGHT,
    })),
    estimateSize: () => ROW_HEIGHT,
    overscan: 2,
    initialRect: { width: DEFAULT_WIDTH, height: DEFAULT_HEIGHT },
  })
  const rows = virtualizer.getVirtualItems()
  const gridTemplateColumns = useMemo(() => `repeat(${columns}, minmax(0, 1fr))`, [columns])

  return (
    <div ref={scrollRef} className="photo-grid" role="grid" aria-label="照片网格">
      <div className="photo-grid__virtual" style={{ height: virtualizer.getTotalSize() }}>
        {rows.map((row) => {
          const rowPhotos = photos.slice(row.index * columns, row.index * columns + columns)
          return (
            <div
              key={row.key}
              className="photo-grid__row"
              role="row"
              style={{ gridTemplateColumns, transform: `translateY(${row.start}px)` }}
            >
              {rowPhotos.map((photo) => {
                const active = photo.id === activeId
                const selected = selectedIds.has(photo.id)
                return (
                  <article
                    key={photo.id}
                    data-photo-id={photo.id}
                    className={`photo-tile${active ? ' is-active' : ''}${selected ? ' is-selected' : ''}`}
                    role="gridcell"
                    aria-selected={selected}
                  >
                    <button
                      type="button"
                      className="photo-tile__preview"
                      aria-label={`查看 ${photo.filename}`}
                      onClick={() => onActivate(photo.id)}
                    >
                      <img src={assetUrl(photo.thumbnail_url)} alt={photo.filename} loading="lazy" />
                    </button>
                    <span className="photo-tile__stars" aria-label={`${photo.stars}星`}>
                      {photo.stars === 0 ? '0' : '★'.repeat(photo.stars)}
                    </span>
                    {photo.needs_review && <CircleAlert className="photo-tile__review" size={16} aria-label="待复核" />}
                    <button
                      type="button"
                      className="photo-tile__select"
                      aria-label={`${selected ? '取消选择' : '选择'} ${photo.filename}`}
                      aria-pressed={selected}
                      onClick={() => onToggleSelected(photo.id)}
                    >
                      <Check size={14} />
                    </button>
                    <footer title={photo.filename}>{photo.filename}</footer>
                  </article>
                )
              })}
            </div>
          )
        })}
      </div>
    </div>
  )
}
