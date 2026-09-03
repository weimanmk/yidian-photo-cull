export type ThemeMode = 'light' | 'dark'

export const THEME_STORAGE_KEY = 'yidian-photo-cull-theme'

export function normalizeTheme(value: string | null | undefined): ThemeMode {
  return value === 'dark' ? 'dark' : 'light'
}

export function loadTheme(): ThemeMode {
  try {
    return normalizeTheme(window.localStorage.getItem(THEME_STORAGE_KEY))
  } catch {
    return 'light'
  }
}

export function applyTheme(theme: ThemeMode) {
  document.documentElement.dataset.theme = theme
  document.documentElement.style.colorScheme = theme
}

export function saveTheme(theme: ThemeMode) {
  applyTheme(theme)
  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, theme)
  } catch {
    // 在存储被系统策略禁用时，主题仍可在当前会话中正常使用。
  }
}
