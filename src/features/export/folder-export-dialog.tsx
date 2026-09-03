import { LoaderCircle } from 'lucide-react'
import { useEffect, useState } from 'react'
import { folderExportApi } from '../../api'
import { Button } from '../../components/ui/button'
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from '../../components/ui/dialog'
import { ToggleGroup, ToggleGroupItem } from '../../components/ui/toggle-group'
import type { ExportPlan, ExportReceipt } from '../../types'
import type { DesktopBridge } from '../lightroom/use-lightroom-operation'

export interface FolderExportApi {
  preflight: (destination: string, projectId: string, minimumStars: 1 | 2 | 3) => Promise<ExportPlan>
  execute: (operationId: string, planHash: string) => Promise<ExportReceipt>
}

interface FolderExportDialogProps {
  projectId: string
  api?: FolderExportApi
  desktop: DesktopBridge | null
  open?: boolean
  onOpenChange?: (open: boolean) => void
}

type ExportPhase = 'selecting' | 'preflighting' | 'awaiting_confirmation' | 'executing' | 'complete' | 'failed'

export default function FolderExportDialog({
  projectId,
  api = folderExportApi,
  desktop,
  open = true,
  onOpenChange,
}: FolderExportDialogProps) {
  const [minimumStars, setMinimumStars] = useState<1 | 2 | 3>(2)
  const [phase, setPhase] = useState<ExportPhase>('selecting')
  const [plan, setPlan] = useState<ExportPlan | null>(null)
  const [receipt, setReceipt] = useState<ExportReceipt | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!open) return
    setPhase('selecting')
    setPlan(null)
    setReceipt(null)
    setError(null)
  }, [open, projectId])

  const chooseDestination = async () => {
    if (!desktop || phase === 'preflighting' || phase === 'executing') return
    const destination = await desktop.selectOutputFolder()
    if (!destination) return
    setPhase('preflighting')
    setError(null)
    try {
      const next = await api.preflight(destination, projectId, minimumStars)
      setPlan(next)
      setPhase('awaiting_confirmation')
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : '导出预检失败')
      setPhase('failed')
    }
  }

  const confirm = async () => {
    if (!plan || phase !== 'awaiting_confirmation') return
    setPhase('executing')
    setError(null)
    try {
      setReceipt(await api.execute(plan.operation_id, plan.plan_hash))
      setPhase('complete')
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : '文件夹导出失败')
      setPhase('failed')
    }
  }

  const changeMinimumStars = (value: string) => {
    if (!value) return
    setMinimumStars(Number(value) as 1 | 2 | 3)
    setPlan(null)
    setReceipt(null)
    setPhase('selecting')
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent aria-describedby={undefined} className="operation-dialog">
        <DialogHeader><DialogTitle>导出文件夹</DialogTitle></DialogHeader>

        <div className="export-rating-row">
          <span>最低星级</span>
          <ToggleGroup type="single" value={String(minimumStars)} onValueChange={changeMinimumStars}>
            <ToggleGroupItem value="1">1星</ToggleGroupItem>
            <ToggleGroupItem value="2">2星</ToggleGroupItem>
            <ToggleGroupItem value="3">3星</ToggleGroupItem>
          </ToggleGroup>
        </div>

        {plan && !receipt && (
          <>
            <div className="operation-catalog"><span>目录</span><b>{plan.destination}</b></div>
            <div className="operation-counts" aria-label="导出预检计数">
              <span>复制 {plan.copy_count}</span>
              <span>跳过 {plan.skip_count}</span>
              <span>冲突 {plan.conflict_count}</span>
              <span className={plan.invalid_count ? 'is-danger' : ''}>无效 {plan.invalid_count}</span>
            </div>
          </>
        )}

        {receipt && (
          <div className="export-receipt">
            <strong>{receipt.destination}</strong>
            <span>已复制 {receipt.copied}</span>
            <span>已跳过 {receipt.skipped}</span>
            <span>冲突 {receipt.conflicts}</span>
            <span>无效 {receipt.invalid}</span>
          </div>
        )}

        {error && <div className="error-row">{error}</div>}

        <DialogFooter>
          {(phase === 'selecting' || phase === 'failed') && (
            <Button disabled={!desktop} onClick={chooseDestination}>选择导出目录</Button>
          )}
          {phase === 'preflighting' && <Button disabled><LoaderCircle className="spin" size={15} />预检中</Button>}
          {phase === 'awaiting_confirmation' && <Button onClick={confirm}>确认导出</Button>}
          {phase === 'executing' && <Button disabled><LoaderCircle className="spin" size={15} />导出中</Button>}
          {phase === 'complete' && <Button variant="outline" onClick={() => onOpenChange?.(false)}>完成</Button>}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
