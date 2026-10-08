import { useMemo, useState } from 'react'
import { assetUrl } from '../../api'
import { Button } from '../../components/ui/button'
import { Dialog, DialogContent, DialogDescription, DialogTitle, DialogTrigger } from '../../components/ui/dialog'
import type { IdentityCorrectionRequest, ScanResults } from '../../types'

const PAGE_SIZE = 48

export default function IdentityCorrectionDialog({ results, onSave }: {
  results: ScanResults
  onSave: (request: IdentityCorrectionRequest) => Promise<void>
}) {
  const [open, setOpen] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [people, setPeople] = useState<string[]>([])
  const [selected, setSelected] = useState<string[]>([])
  const [filter, setFilter] = useState('all')
  const [page, setPage] = useState(0)
  const faces = useMemo(() => results.photos.flatMap(photo => (photo.faces ?? []).map(face => ({
    photo, face, key: JSON.stringify([photo.id, face.face_id]),
  }))), [results.photos])
  const identities = useMemo(() => [...new Set(faces.flatMap(({ face }) => face.person_id ? [face.person_id] : []))].sort(), [faces])
  const visible = faces.filter(({ face }) => filter === 'all' || (filter === 'unknown' ? !face.person_id : face.person_id === filter))
  const toggle = (items: string[], key: string) => items.includes(key) ? items.filter(item => item !== key) : [...items, key]

  async function save(operation: 'merge' | 'split') {
    setBusy(true)
    setError('')
    try {
      await onSave({ operation, revision: results.identity_revision ?? 0,
        person_ids: operation === 'merge' ? people : [],
        faces: operation === 'split' ? faces.filter(item => selected.includes(item.key)).map(({ photo, face }) => ({ photo_id: photo.id, face_id: face.face_id })) : [],
      })
      setOpen(false)
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : '保存失败，请重试')
    } finally {
      setBusy(false)
    }
  }

  return <Dialog open={open} onOpenChange={next => {
    if (busy) return
    setOpen(next)
    if (next) { setPeople([]); setSelected([]); setError(''); setFilter('all'); setPage(0) }
  }}>
    <DialogTrigger asChild><Button variant="outline" size="sm">修正人物身份</Button></DialogTrigger>
    <DialogContent className="max-h-[90vh] overflow-y-auto">
      <DialogTitle>修正人物身份</DialogTitle>
      <DialogDescription>合并同一人的多个身份，或勾选人脸拆分为新人物。同一照片中的不同人脸不能合并。保存后请重新扫描，更新优选和覆盖结果；人工星级会保留。</DialogDescription>
      <fieldset disabled={busy} className="grid gap-3">
        <legend>合并人物</legend>
        <div className="flex max-h-32 flex-wrap gap-3 overflow-y-auto">
          {identities.map(person => <label key={person} className="flex items-center gap-1">
            <input type="checkbox" aria-label={`选择人物 ${person}`} checked={people.includes(person)} onChange={() => setPeople(toggle(people, person))} />{person}
          </label>)}
          {!identities.length && <span>暂无已识别人物，可从下方人脸创建身份。</span>}
        </div>
        <Button disabled={busy || people.length < 2} onClick={() => void save('merge')}>{busy ? '正在保存…' : '合并所选人物'}</Button>
      </fieldset>
      <fieldset disabled={busy} className="grid gap-3">
        <legend>拆分人脸（已选 {selected.length}）</legend>
        <label>查看人物 <select aria-label="查看待修正人物" value={filter} onChange={event => { setFilter(event.target.value); setPage(0) }}>
          <option value="all">全部人物</option><option value="unknown">未识别</option>
          {identities.map(person => <option key={person} value={person}>{person}</option>)}
        </select></label>
        <div className="grid grid-cols-3 gap-2">
          {visible.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE).map(({ photo, face, key }) => <label key={key} className="min-w-0 text-sm">
            <span className="relative block">
              <img src={assetUrl(photo.thumbnail_url)} alt="" className="block w-full" loading="lazy" />
              <span className="absolute border-2 border-primary" style={{ left: `${face.bbox[0] * 100}%`, top: `${face.bbox[1] * 100}%`, width: `${(face.bbox[2] - face.bbox[0]) * 100}%`, height: `${(face.bbox[3] - face.bbox[1]) * 100}%` }} />
            </span>
            <input type="checkbox" aria-label={`选择人脸 ${photo.filename} · ${face.face_id}`} checked={selected.includes(key)} onChange={() => setSelected(toggle(selected, key))} />
            <span className="block truncate" title={photo.filename}>{photo.filename}</span><span>{face.person_id ?? '未识别'}</span>
          </label>)}
        </div>
        {!visible.length && <p>暂无人脸</p>}
        {visible.length > PAGE_SIZE && <div className="flex items-center justify-between">
          <Button variant="outline" disabled={page === 0} onClick={() => setPage(page - 1)}>上一页</Button>
          <span>{page + 1} / {Math.ceil(visible.length / PAGE_SIZE)}</span>
          <Button variant="outline" disabled={(page + 1) * PAGE_SIZE >= visible.length} onClick={() => setPage(page + 1)}>下一页</Button>
        </div>}
        <Button disabled={busy || !selected.length} onClick={() => void save('split')}>拆分所选人脸为新人物</Button>
      </fieldset>
      {error && <p role="alert">{error}</p>}
    </DialogContent>
  </Dialog>
}
