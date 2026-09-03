import type { EngineSettings, ExportPlan, ExportReceipt, HealthResponse, LightroomBackendStatus, LightroomOperation, PhotoStars, ProjectSummary, ScanResults, ScanStatus } from './types'

let baseUrl = 'http://127.0.0.1:8767'
let apiToken = ''

export async function initializeApi(): Promise<string> {
  if (window.desktop) {
    const info = await window.desktop.backendInfo()
    baseUrl = info.url || baseUrl
    apiToken = info.token || ''
  }
  return baseUrl
}

async function request<T>(pathname: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${baseUrl}${pathname}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(apiToken ? { 'X-PhotoCull-Token': apiToken } : {}),
      ...(init?.headers ?? {}),
    },
  })
  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(detail.detail || `请求失败 (${response.status})`)
  }
  return response.json() as Promise<T>
}

export const api = {
  health: () => request<HealthResponse>('/api/health'),
  status: () => request<ScanStatus>('/api/scan/status'),
  results: () => request<ScanResults>('/api/scan/results'),
  projects: () => request<ProjectSummary[]>('/api/projects'),
  loadProject: (projectId: string) => request<ScanResults>(`/api/projects/${encodeURIComponent(projectId)}`),
  settings: () => request<EngineSettings>('/api/settings'),
  updateSettings: (settings: Partial<EngineSettings>) => request<EngineSettings>('/api/settings', {
    method: 'PATCH',
    body: JSON.stringify(settings),
  }),
  startScan: (folder: string, settings: Pick<EngineSettings, 'grouping_preset' | 'keep_per_group' | 'coverage_enabled' | 'coverage_window_minutes' | 'recursive'>) => request<ScanStatus>('/api/scan/start', {
    method: 'POST',
    body: JSON.stringify({ folder, ...settings }),
  }),
  cancelScan: () => request<ScanStatus>('/api/scan/cancel', { method: 'POST' }),
  ratePhoto: (photoId: string, stars: PhotoStars, locked = true) => request<PhotoResultResponse>(`/api/photos/${encodeURIComponent(photoId)}/rating`, {
    method: 'PATCH',
    body: JSON.stringify({ stars, locked }),
  }),
  lightroomStatus: () => request<LightroomBackendStatus>('/api/lightroom/status'),
  createLightroomPreflight: (projectId: string) => request<LightroomOperation>('/api/lightroom/preflights', {
    method: 'POST',
    body: JSON.stringify({ project_id: projectId }),
  }),
  getLightroomOperation: (operationId: string) => request<LightroomOperation>(`/api/lightroom/operations/${encodeURIComponent(operationId)}`),
  executeLightroomOperation: (operationId: string) => request<LightroomOperation>(`/api/lightroom/operations/${encodeURIComponent(operationId)}/execute`, {
    method: 'POST',
  }),
  preflightExport: (destination: string, projectId: string, minimumStars: 1 | 2 | 3) => request<ExportPlan>('/api/exports/preflights', {
    method: 'POST',
    body: JSON.stringify({ destination, project_id: projectId, minimum_stars: minimumStars }),
  }),
  executeExport: (operationId: string, planHash: string) => request<ExportReceipt>(`/api/exports/${encodeURIComponent(operationId)}/execute`, {
    method: 'POST',
    body: JSON.stringify({ plan_hash: planHash, confirmed: true }),
  }),
}

interface PhotoResultResponse {
  ok: boolean
}

export function assetUrl(pathname: string): string {
  const separator = pathname.includes('?') ? '&' : '?'
  return `${baseUrl}${pathname}${apiToken ? `${separator}token=${encodeURIComponent(apiToken)}` : ''}`
}

export const lightroomOperationApi = {
  getStatus: api.lightroomStatus,
  createPreflight: api.createLightroomPreflight,
  getOperation: api.getLightroomOperation,
  execute: api.executeLightroomOperation,
}

export const folderExportApi = {
  preflight: api.preflightExport,
  execute: api.executeExport,
}
