"""Face-level user decisions, stored atomically with their source-scoped project."""
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4


class CorrectionStore:
    def __init__(self, projects):
        self.projects = projects

    def load(self, source_root: str) -> dict:
        canonical = str(Path(source_root).resolve())
        for path in sorted(self.projects.root.glob('*.json'), key=lambda p: p.stat().st_mtime_ns, reverse=True):
            try:
                payload = json.loads(path.read_text(encoding='utf-8'))
                result = payload['results']
                source = result.get('source_root')
                if not source:
                    # v0.2.1 saved absolute file paths and relative paths, but no
                    # source_root. Require their complete agreement with the
                    # requested root; basename/common-parent guesses are unsafe.
                    evidence = [(p.get('relative_path'), payload.get('files', {}).get(p['id']))
                                for p in result.get('photos', [])]
                    if evidence and all(relative and absolute
                            and not Path(relative).is_absolute()
                            and '..' not in Path(relative).parts
                            and (Path(canonical) / relative).resolve() == Path(absolute).resolve()
                            for relative, absolute in evidence):
                        source = canonical
                if source != canonical:
                    continue
                state = {key: result.get(key, default) for key, default in (
                    ('identity_corrections', []), ('identity_revision', 0), ('manual_ratings', {}))}
                # Older projects saved manual locks directly on the photos.
                for photo in result.get('photos', []):
                    if photo.get('rating_locked') and photo.get('stars') in (0, 1, 2, 3):
                        state['manual_ratings'].setdefault(photo['id'], {
                            'stars': photo['stars'], 'rating_locked': True,
                        })
                return state
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return {}


def correct_faces(results: dict, operation: str, person_ids: list[str], refs: list[dict]) -> None:
    faces = {(p['id'], f['face_id']): f for p in results['photos'] for f in p.get('faces', [])}
    known = {f.get('person_id') for f in faces.values()} - {None}
    selected = {(ref['photo_id'], ref['face_id']) for ref in refs}
    if operation == 'merge':
        if len(set(person_ids)) < 2 or not set(person_ids) <= known:
            raise ValueError('请选择至少两个当前有效的人物身份')
        selected = {key for key, face in faces.items() if face.get('person_id') in person_ids}
    elif operation == 'split':
        if not selected or not selected <= faces.keys():
            raise ValueError('人脸引用已失效，请刷新项目后重试')
    else:
        raise ValueError('未知身份修正操作')
    if len({key[0] for key in selected}) != len(selected):
        raise ValueError('同一张照片中的不同人脸不能合为同一人物')

    # Freeze the remainder too: a later automatic cluster must not undo a split.
    affected = {faces[key].get('person_id') for key in selected} - {None}
    replacements = {person: 'manual-' + uuid4().hex[:16] for person in affected}
    target = 'manual-' + uuid4().hex[:16]
    anchors = {(a['photo_id'], tuple(a['bbox'])): a for a in results.get('identity_corrections', [])}
    for key, face in faces.items():
        if key in selected or face.get('person_id') in affected:
            # A new explicit decision supersedes the prior anchor even if a
            # later detector moved its box. Keep ambiguous old anchors for review.
            for anchor_key, old in list(anchors.items()):
                if old['photo_id'] != key[0]:
                    continue
                matches = [k for k, f in faces.items()
                           if k[0] == key[0] and _overlap(f['bbox'], old['bbox']) >= .8]
                if matches == [key]:
                    del anchors[anchor_key]
            face['person_id'] = target if key in selected else replacements[face['person_id']]
            anchor = {'photo_id': key[0], 'bbox': list(face['bbox']), 'person_id': face['person_id']}
            anchors[(key[0], tuple(face['bbox']))] = anchor
    results['identity_corrections'] = list(anchors.values())
    for photo in results['photos']:
        photo['person_ids'] = sorted({f['person_id'] for f in photo.get('faces', []) if f.get('person_id')})


def _overlap(left, right) -> float:
    intersection = max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(0, min(left[3], right[3]) - max(left[1], right[1]))
    union = (left[2] - left[0]) * (left[3] - left[1]) + (right[2] - right[0]) * (right[3] - right[1]) - intersection
    return intersection / union if union > 0 else 0


def replay_corrections(photos, anchors: list[dict]) -> list[str]:
    """Conservative geometry matching survives detector order changes; never guess."""
    by_id = {p.id: p for p in photos}
    warnings = []
    assignments = {}
    for anchor in anchors:
        photo = by_id.get(anchor['photo_id'])
        candidates = [] if photo is None else [f for f in photo.faces if _overlap(f.bbox, anchor['bbox']) >= .8]
        if len(candidates) != 1:
            warnings.append('部分人工人脸修正无法可靠匹配，请复核人物身份')
            continue
        key = (photo.id, candidates[0].face_id)
        if key in assignments:
            warnings.append('人工人脸修正匹配存在歧义，请复核人物身份')
            assignments[key] = None
        else:
            assignments[key] = anchor['person_id']
    for photo in photos:
        proposed = [assignments.get((photo.id, f.face_id)) or f.person_id for f in photo.faces]
        counts = {person: proposed.count(person) for person in proposed if person}
        for face, person in zip(photo.faces, proposed):
            if person and counts[person] > 1 and (photo.id, face.face_id) in assignments:
                warnings.append('人工身份修正在同一照片中冲突，请复核')
                continue
            face.person_id = person
        photo.person_ids = sorted({f.person_id for f in photo.faces if f.person_id})
    return sorted(set(warnings))


def replay_manual_ratings(photos, ratings: dict) -> None:
    from .rating_types import TIER_BY_STAR
    for photo in photos:
        rating = ratings.get(photo.id)
        if rating is None or not rating.get('rating_locked'):
            continue
        photo.stars = int(rating['stars'])
        photo.rating_tier = TIER_BY_STAR[photo.stars].value
        photo.rating_origin = 'manual'
        photo.rating_reason = 'manual_override'
        photo.rating_locked = True
