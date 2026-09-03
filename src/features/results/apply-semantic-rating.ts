import type { PhotoResult, PhotoStars, RatingTier, ScanResults } from '../../types'

const tierByStars: Record<PhotoStars, RatingTier> = {
  0: 'waste',
  1: 'valuable',
  2: 'coverage',
  3: 'primary',
}

function duplicateLeaks(photos: PhotoResult[]): number {
  const counts = new Map<string, number>()
  for (const photo of photos) {
    if (photo.stars !== 3 || !photo.strict_duplicate_cluster_id) continue
    const clusterId = photo.strict_duplicate_cluster_id
    counts.set(clusterId, (counts.get(clusterId) ?? 0) + 1)
  }
  return Array.from(counts.values()).reduce((total, count) => total + Math.max(0, count - 1), 0)
}

export function applySemanticRating(
  results: ScanResults,
  photoId: string,
  stars: PhotoStars,
  locked: boolean,
): ScanResults {
  if (!results.photos.some((photo) => photo.id === photoId)) return results

  const photos = results.photos.map((photo): PhotoResult => photo.id === photoId ? {
    ...photo,
    stars,
    rating_tier: tierByStars[stars],
    rating_origin: 'manual',
    rating_reason: 'manual_override',
    rating_locked: locked,
    needs_review: false,
    is_best_pick: stars >= 2,
    coverage_protected: false,
    coverage_person_ids: [],
  } : photo)
  const count = (value: PhotoStars) => photos.filter((photo) => photo.stars === value).length

  return {
    ...results,
    photos,
    summary: {
      ...results.summary,
      selected: photos.filter((photo) => photo.stars >= 2).length,
      issues: photos.filter((photo) => (
        !['selected', 'duplicate'].includes(photo.category)
        || (photo.coverage_protected && photo.issues.length > 0)
      )).length,
      coverage_protected: photos.filter((photo) => photo.coverage_protected).length,
      primary_duplicate_leaks: duplicateLeaks(photos),
      primary: count(3),
      coverage: count(2),
      valuable: count(1),
      waste: count(0),
      stars_0: count(0),
      stars_1: count(1),
      stars_2: count(2),
      stars_3: count(3),
    },
  }
}
