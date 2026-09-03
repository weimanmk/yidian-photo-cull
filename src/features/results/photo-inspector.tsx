import { CircleAlert, PanelRightClose, PanelRightOpen } from 'lucide-react'
import { Button } from '../../components/ui/button'
import { Separator } from '../../components/ui/separator'
import type { PhotoResult, PhotoStars, RatingReason, RatingTier } from '../../types'
import RatingControl from './rating-control'

interface PhotoInspectorProps {
  photo: PhotoResult | null
  collapsed: boolean
  onCollapsedChange: (collapsed: boolean) => void
  onRate: (photoId: string, stars: PhotoStars, locked: true) => void | Promise<void>
}

const tierLabels: Record<RatingTier, string> = {
  waste: '废片',
  valuable: '有价值',
  coverage: '人物×环节补位',
  primary: '精选',
}

const reasonLabels: Partial<Record<RatingReason, string>> = {
  primary_rank: '组内首选',
  person_stage_gap: '覆盖缺口',
  person_stage_reserve: '覆盖备选',
  unique_moment: '独特瞬间',
  technical_reject: '技术废片',
  redundant_reject: '重复淘汰',
  manual_override: '人工调整',
  legacy_score: '旧项目迁移',
}

function QualityRow({ label, value }: { label: string; value: number }) {
  const percent = Math.round(Math.max(0, Math.min(1, value)) * 100)
  return (
    <div className="quality-row">
      <span>{label}</span>
      <div><i style={{ width: `${percent}%` }} /></div>
      <b>{percent}</b>
    </div>
  )
}

export default function PhotoInspector({ photo, collapsed, onCollapsedChange, onRate }: PhotoInspectorProps) {
  if (collapsed) {
    return (
      <aside className="photo-inspector photo-inspector--collapsed">
        <Button variant="ghost" size="icon" aria-label="展开检查器" onClick={() => onCollapsedChange(false)}>
          <PanelRightOpen size={17} />
        </Button>
      </aside>
    )
  }

  return (
    <aside className="photo-inspector">
      <header>
        <strong>照片信息</strong>
        <Button variant="ghost" size="icon" aria-label="收起检查器" onClick={() => onCollapsedChange(true)}>
          <PanelRightClose size={17} />
        </Button>
      </header>

      {photo && (
        <div className="photo-inspector__content">
          <div className="inspector-filename" title={photo.filename}>{photo.filename}</div>
          <RatingControl
            value={photo.stars}
            locked={photo.rating_locked}
            onRate={(stars, locked) => onRate(photo.id, stars, locked)}
          />
          <dl className="inspector-facts">
            <div><dt>级别</dt><dd>{tierLabels[photo.rating_tier]}</dd></div>
            <div><dt>原因</dt><dd>{reasonLabels[photo.rating_reason as RatingReason] ?? photo.rating_reason}</dd></div>
            <div><dt>环节</dt><dd>{photo.stage_label || '未识别'}</dd></div>
            <div><dt>人物</dt><dd>{photo.person_ids.join('、') || '未识别'}</dd></div>
          </dl>
          {photo.needs_review && <div className="inspector-review"><CircleAlert size={15} />待复核</div>}
          <Separator />
          <section className="quality-list" aria-label="质量指标">
            <QualityRow label="清晰" value={photo.metrics.sharpness_score} />
            <QualityRow label="人脸" value={photo.metrics.face_sharpness_score} />
            <QualityRow label="曝光" value={photo.metrics.exposure_score} />
            <QualityRow label="构图" value={photo.metrics.composition_score} />
            <QualityRow label="眼睛" value={photo.metrics.eye_score} />
          </section>
        </div>
      )}
    </aside>
  )
}
