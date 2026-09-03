import { useEffect } from 'react'
import type { PhotoStars } from '../../types'
import type { ResultsWorkspaceMode } from './use-results-workspace'

interface CullingShortcutOptions {
  enabled: boolean
  onRate: (stars: PhotoStars) => void
  onNext: () => void
  onPrevious: () => void
  onModeChange: (mode: ResultsWorkspaceMode) => void
}

function isEditingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  if (target.isContentEditable || target.closest('[contenteditable="true"]')) return true
  return ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)
}

export default function useCullingShortcuts({
  enabled,
  onRate,
  onNext,
  onPrevious,
  onModeChange,
}: CullingShortcutOptions) {
  useEffect(() => {
    if (!enabled) return

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey) return
      if (isEditingTarget(event.target) || document.querySelector('[role="dialog"]')) return

      if (/^[0-3]$/.test(event.key)) {
        event.preventDefault()
        onRate(Number(event.key) as PhotoStars)
        if (event.shiftKey) onNext()
        return
      }

      if (event.key === 'ArrowLeft') {
        event.preventDefault()
        onPrevious()
        return
      }
      if (event.key === 'ArrowRight') {
        event.preventDefault()
        onNext()
        return
      }

      const modeByKey: Partial<Record<string, ResultsWorkspaceMode>> = {
        g: 'grid',
        e: 'loupe',
        c: 'compare',
      }
      const mode = modeByKey[event.key.toLowerCase()]
      if (mode) {
        event.preventDefault()
        onModeChange(mode)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [enabled, onModeChange, onNext, onPrevious, onRate])
}
