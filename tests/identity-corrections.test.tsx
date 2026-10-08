import React from 'react'
import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import ResultsPage from '../src/pages/ResultsPage'
import type { ScanResults } from '../src/types'

const results = {
  project_id: 'project', project_name: 'Test', identity_revision: 4,
  photos: [
    { id: 'a', filename: 'a.jpg', person_ids: ['p1'], faces: [{ face_id: 'a0', person_id: 'p1', bbox: [0, 0, .3, .3] }], stars: 0, rating_tier: 'waste', rating_reason: 'redundant_reject', metrics: {}, thumbnail_url: '/a', image_url: '/a' },
    { id: 'b', filename: 'b.jpg', person_ids: ['p2'], faces: [{ face_id: 'b0', person_id: 'p2', bbox: [.4, .4, .7, .7] }], stars: 3, rating_tier: 'primary', metrics: {}, thumbnail_url: '/b', image_url: '/b' },
  ], summary: { total: 2 }, groups: [],
} as unknown as ScanResults
const base = { onRate: vi.fn(), onImportLightroom: vi.fn(), onExportFolder: vi.fn() }

describe('identity corrections', () => {
  it('sends a versioned merge and shows a pending state', async () => {
    let finish!: () => void
    const onCorrectIdentities = vi.fn(() => new Promise<void>((resolve) => { finish = resolve }))
    render(<ResultsPage {...base} results={results} onCorrectIdentities={onCorrectIdentities} />)
    fireEvent.click(screen.getByRole('button', { name: '修正人物身份' }))
    fireEvent.click(screen.getByLabelText('选择人物 p1'))
    fireEvent.click(screen.getByLabelText('选择人物 p2'))
    fireEvent.click(screen.getByRole('button', { name: '合并所选人物' }))
    expect(onCorrectIdentities).toHaveBeenCalledWith({ operation: 'merge', person_ids: ['p1', 'p2'], faces: [], revision: 4 })
    expect(screen.getByRole('button', { name: '正在保存…' })).toBeDisabled()
    finish()
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument())
  })
  it('splits selected face refs and keeps an actionable error visible', async () => {
    const onCorrectIdentities = vi.fn().mockRejectedValue(new Error('人物身份已更新，请刷新后重试'))
    render(<ResultsPage {...base} results={results} onCorrectIdentities={onCorrectIdentities} />)
    fireEvent.click(screen.getByRole('button', { name: '修正人物身份' }))
    fireEvent.click(screen.getByLabelText('选择人脸 a.jpg · a0'))
    fireEvent.click(screen.getByRole('button', { name: '拆分所选人脸为新人物' }))
    await screen.findByRole('alert')
    expect(onCorrectIdentities).toHaveBeenCalledWith({ operation: 'split', person_ids: [], faces: [{ photo_id: 'a', face_id: 'a0' }], revision: 4 })
    expect(screen.getByRole('alert')).toHaveTextContent('请刷新后重试')
  })
  it('labels zero-star as unselected and displays stale report warning', () => {
    render(<ResultsPage {...base} results={{ ...results, identity_rescan_required: true }} />)
    expect(screen.getByText('未入选')).toBeInTheDocument()
    expect(screen.getByText('重复淘汰')).toBeInTheDocument()
    expect(screen.getByRole('status')).toHaveTextContent('重新扫描')
  })
})
