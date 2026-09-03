import { useCallback, useEffect, useMemo, useState } from 'react'
import type { LightroomBackendStatus, LightroomOperation } from '../../types'

export type LightroomDialogPhase =
  | 'checking_plugin'
  | 'install_required'
  | 'waiting_for_lightroom'
  | 'preflighting'
  | 'awaiting_confirmation'
  | 'executing'
  | 'complete'
  | 'failed'

export type DesktopBridge = NonNullable<Window['desktop']>
export type DesktopLightroomStatus = Awaited<ReturnType<DesktopBridge['lightroomStatus']>>

export interface LightroomOperationApi {
  getStatus: () => Promise<LightroomBackendStatus>
  createPreflight: (projectId: string) => Promise<LightroomOperation>
  getOperation: (operationId: string) => Promise<LightroomOperation>
  execute: (operationId: string) => Promise<LightroomOperation>
}

interface UseLightroomOperationOptions {
  projectId: string
  api: LightroomOperationApi
  desktop: DesktopBridge | null
  enabled: boolean
}

const POLLABLE_STATUSES = new Set(['waiting_for_plugin', 'preflighting', 'executing', 'verifying'])

function isFreshHeartbeat(status: LightroomBackendStatus | null): boolean {
  const heartbeat = status?.plugin_heartbeat
  if (!heartbeat?.running || heartbeat.plugin_id !== 'com.yidian.photocull.lightroom') return false
  const timestamp = Date.parse(heartbeat.timestamp)
  return Number.isFinite(timestamp) && Math.abs(Date.now() - timestamp) <= 15_000
}

function phaseForOperation(operation: LightroomOperation): LightroomDialogPhase {
  if (operation.status === 'awaiting_confirmation') return 'awaiting_confirmation'
  if (operation.status === 'complete') return 'complete'
  if (['executing', 'verifying'].includes(operation.status)) return 'executing'
  if (['waiting_for_plugin', 'preflighting', 'created'].includes(operation.status)) return 'preflighting'
  return 'failed'
}

export function useLightroomOperation({ projectId, api, desktop, enabled }: UseLightroomOperationOptions) {
  const [phase, setPhase] = useState<LightroomDialogPhase>('checking_plugin')
  const [desktopStatus, setDesktopStatus] = useState<DesktopLightroomStatus | null>(null)
  const [backendStatus, setBackendStatus] = useState<LightroomBackendStatus | null>(null)
  const [operation, setOperation] = useState<LightroomOperation | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const heartbeatReady = useMemo(() => isFreshHeartbeat(backendStatus), [backendStatus])

  const applyOperation = useCallback((next: LightroomOperation) => {
    setOperation(next)
    setPhase(phaseForOperation(next))
    if (next.error_message) setError(next.error_message)
    else if (next.status === 'pending_rating') setError(`${next.counts?.pending_rating ?? 0} 张星级待处理`)
  }, [])

  const checkPlugin = useCallback(async () => {
    if (!desktop) {
      setError('仅桌面端可导入 Lightroom')
      setPhase('failed')
      return
    }
    setBusy(true)
    setError(null)
    setPhase('checking_plugin')
    try {
      const [nextDesktopStatus, nextBackendStatus] = await Promise.all([
        desktop.lightroomStatus(),
        api.getStatus(),
      ])
      setDesktopStatus(nextDesktopStatus)
      setBackendStatus(nextBackendStatus)
      setPhase(nextDesktopStatus.plugin.compatible ? 'waiting_for_lightroom' : 'install_required')
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Lightroom 状态检查失败')
      setPhase('failed')
    } finally {
      setBusy(false)
    }
  }, [api, desktop])

  useEffect(() => {
    if (!enabled) return
    setOperation(null)
    void checkPlugin()
  }, [checkPlugin, enabled, projectId])

  useEffect(() => {
    if (!enabled || phase !== 'waiting_for_lightroom' || heartbeatReady) return
    const timer = window.setInterval(() => {
      void api.getStatus().then(setBackendStatus).catch(() => undefined)
    }, 1_000)
    return () => window.clearInterval(timer)
  }, [api, enabled, heartbeatReady, phase])

  useEffect(() => {
    if (!enabled || !operation || !POLLABLE_STATUSES.has(operation.status)) return
    const timer = window.setInterval(() => {
      void api.getOperation(operation.id).then(applyOperation).catch((cause) => {
        setError(cause instanceof Error ? cause.message : 'Lightroom 操作刷新失败')
        setPhase('failed')
      })
    }, 750)
    return () => window.clearInterval(timer)
  }, [api, applyOperation, enabled, operation])

  const installPlugin = useCallback(async () => {
    if (!desktop || busy) return
    setBusy(true)
    setError(null)
    try {
      await desktop.installLightroomPlugin()
      await desktop.launchLightroom()
      await checkPlugin()
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Lightroom 插件安装失败')
      setPhase('failed')
    } finally {
      setBusy(false)
    }
  }, [busy, checkPlugin, desktop])

  const launchLightroom = useCallback(async () => {
    if (!desktop || busy) return
    setBusy(true)
    setError(null)
    try {
      await desktop.launchLightroom()
      setPhase('waiting_for_lightroom')
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Lightroom 启动失败')
      setPhase('failed')
    } finally {
      setBusy(false)
    }
  }, [busy, desktop])

  const startPreflight = useCallback(async () => {
    if (busy || !heartbeatReady) return
    setBusy(true)
    setError(null)
    setPhase('preflighting')
    try {
      applyOperation(await api.createPreflight(projectId))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Lightroom 预检失败')
      setPhase('failed')
    } finally {
      setBusy(false)
    }
  }, [api, applyOperation, busy, heartbeatReady, projectId])

  const confirm = useCallback(async () => {
    if (busy || !operation?.can_execute || (operation.counts?.invalid ?? 0) > 0) return
    setBusy(true)
    setError(null)
    setPhase('executing')
    try {
      applyOperation(await api.execute(operation.id))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Lightroom 导入失败')
      setPhase('failed')
    } finally {
      setBusy(false)
    }
  }, [api, applyOperation, busy, operation])

  return {
    phase,
    desktopStatus,
    backendStatus,
    heartbeatReady,
    operation,
    error,
    busy,
    checkPlugin,
    installPlugin,
    launchLightroom,
    startPreflight,
    confirm,
  }
}
