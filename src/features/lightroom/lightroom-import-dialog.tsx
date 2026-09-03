import { LoaderCircle } from 'lucide-react'
import { lightroomOperationApi } from '../../api'
import { Button } from '../../components/ui/button'
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from '../../components/ui/dialog'
import type { LightroomReceiptCounts } from '../../types'
import { useLightroomOperation, type DesktopBridge, type LightroomOperationApi } from './use-lightroom-operation'

interface LightroomImportDialogProps {
  projectId: string
  api?: LightroomOperationApi
  desktop: DesktopBridge | null
  open?: boolean
  onOpenChange?: (open: boolean) => void
}

function ReceiptCounts({ counts }: { counts: LightroomReceiptCounts }) {
  return (
    <div className="operation-counts" aria-label="Lightroom 预检计数">
      <span>总计 {counts.total}</span>
      <span>新增 {counts.new}</span>
      <span>更新 {counts.update}</span>
      <span>不变 {counts.unchanged}</span>
      <span>保护 {counts.protected}</span>
      <span className={counts.invalid ? 'is-danger' : ''}>无效 {counts.invalid}</span>
    </div>
  )
}

export default function LightroomImportDialog({
  projectId,
  api = lightroomOperationApi,
  desktop,
  open = true,
  onOpenChange,
}: LightroomImportDialogProps) {
  const workflow = useLightroomOperation({ projectId, api, desktop, enabled: open })
  const counts = workflow.operation?.counts
  const remainingCount = counts ? Math.max(0, counts.new + counts.update - counts.verified) : 0
  const failedCount = counts ? Math.max(0, remainingCount - counts.pending_rating) : 0

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent aria-describedby={undefined} className="operation-dialog">
        <DialogHeader>
          <DialogTitle>导入 Lightroom</DialogTitle>
        </DialogHeader>

        <div className="operation-status-row">
          <span>插件</span>
          <b>{workflow.desktopStatus?.plugin.compatible ? '已就绪' : '待安装'}</b>
          <span>心跳</span>
          <b>{workflow.heartbeatReady ? '正常' : '等待'}</b>
        </div>

        {workflow.operation?.catalog_name && (
          <div className="operation-catalog"><span>目录</span><b>{workflow.operation.catalog_name}</b></div>
        )}

        {counts && <ReceiptCounts counts={counts} />}

        {(workflow.phase === 'executing' || workflow.phase === 'complete') && counts && (
          <div className="operation-progress-counts">
            <span>已验证 {counts.verified}</span>
            <span>{workflow.phase === 'executing' ? `剩余 ${remainingCount}` : `待处理 ${counts.pending_rating}`}</span>
            <span>{workflow.phase === 'executing' ? `待处理 ${counts.pending_rating}` : `失败 ${failedCount}`}</span>
          </div>
        )}

        {workflow.phase === 'awaiting_confirmation' && (
          <div className="operation-safety">新加入的照片撤销时仍保留在 Lightroom 目录中</div>
        )}

        {workflow.error && <div className="error-row">{workflow.error}</div>}

        <DialogFooter>
          {workflow.phase === 'checking_plugin' && <Button disabled><LoaderCircle className="spin" size={15} />检查中</Button>}
          {workflow.phase === 'install_required' && <Button disabled={workflow.busy} onClick={workflow.installPlugin}>安装插件</Button>}
          {workflow.phase === 'waiting_for_lightroom' && !workflow.heartbeatReady && (
            <>
              <Button variant="outline" disabled={workflow.busy} onClick={workflow.checkPlugin}>重新检查</Button>
              <Button disabled={workflow.busy || !workflow.desktopStatus?.lightroom.found} onClick={workflow.launchLightroom}>启动 Lightroom</Button>
            </>
          )}
          {workflow.phase === 'waiting_for_lightroom' && workflow.heartbeatReady && (
            <Button disabled={workflow.busy} onClick={workflow.startPreflight}>开始预检</Button>
          )}
          {workflow.phase === 'preflighting' && <Button disabled><LoaderCircle className="spin" size={15} />预检中</Button>}
          {workflow.phase === 'awaiting_confirmation' && (
            <Button
              disabled={workflow.busy || !workflow.operation?.can_execute || (counts?.invalid ?? 0) > 0}
              onClick={workflow.confirm}
            >
              确认导入
            </Button>
          )}
          {workflow.phase === 'executing' && <Button disabled><LoaderCircle className="spin" size={15} />执行中</Button>}
          {workflow.phase === 'failed' && <Button variant="outline" disabled={workflow.busy} onClick={workflow.checkPlugin}>重新检查</Button>}
          {workflow.phase === 'complete' && <Button variant="outline" onClick={() => onOpenChange?.(false)}>完成</Button>}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
