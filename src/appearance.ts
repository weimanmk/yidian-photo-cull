export type ThemeMode = 'light' | 'dark' | 'system'
export type UiScale = 1 | 1.1 | 1.25

export interface Appearance {
  theme: ThemeMode
  scale: UiScale
}

export const UI_SCALES: readonly UiScale[] = [1, 1.1, 1.25]

export function stepUiScale(current: UiScale, direction: -1 | 1): UiScale {
  const index = UI_SCALES.indexOf(current)
  const nextIndex = Math.max(0, Math.min(UI_SCALES.length - 1, index + direction))
  return UI_SCALES[nextIndex]
}

export const APPEARANCE_STORAGE_KEY = 'yidian-photo-cull-appearance'

const DEFAULT_APPEARANCE: Appearance = {
  theme: 'light',
  scale: 1.1,
}

function isThemeMode(value: unknown): value is ThemeMode {
  return value === 'light' || value === 'dark' || value === 'system'
}

function isUiScale(value: unknown): value is UiScale {
  return value === 1 || value === 1.1 || value === 1.25
}

export function normalizeAppearance(value: Partial<Appearance>): Appearance {
  return {
    theme: isThemeMode(value.theme) ? value.theme : DEFAULT_APPEARANCE.theme,
    scale: isUiScale(value.scale) ? value.scale : DEFAULT_APPEARANCE.scale,
  }
}

export function resolveTheme(mode: ThemeMode, systemDark: boolean): 'light' | 'dark' {
  if (mode === 'system') {
    return systemDark ? 'dark' : 'light'
  }
  return mode
}

export function loadAppearance(): Appearance {
  try {
    const saved = window.localStorage.getItem(APPEARANCE_STORAGE_KEY)
    if (!saved) {
      return { ...DEFAULT_APPEARANCE }
    }
    return normalizeAppearance(JSON.parse(saved) as Partial<Appearance>)
  } catch {
    return { ...DEFAULT_APPEARANCE }
  }
}

export function applyAppearance(value: Appearance): void {
  const systemDark = window.matchMedia('(prefers-color-scheme: dark)').matches
  const resolvedTheme = resolveTheme(value.theme, systemDark)
  document.documentElement.dataset.theme = resolvedTheme
  document.documentElement.style.colorScheme = resolvedTheme
  document.documentElement.style.setProperty('--ui-scale', String(value.scale))
}

export function saveAppearance(value: Appearance): void {
  applyAppearance(value)
  try {
    window.localStorage.setItem(APPEARANCE_STORAGE_KEY, JSON.stringify(value))
  } catch {
    // 在存储被系统策略禁用时，外观仍可在当前会话中正常使用。
  }
}
