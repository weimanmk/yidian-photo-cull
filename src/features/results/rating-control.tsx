import { Lock, LockOpen } from 'lucide-react'
import { Button } from '../../components/ui/button'
import type { PhotoStars } from '../../types'

interface RatingControlProps {
  value: PhotoStars
  locked: boolean
  onRate: (stars: PhotoStars, locked: true) => void | Promise<void>
  disabled?: boolean
}

const ratings: PhotoStars[] = [0, 1, 2, 3]

export default function RatingControl({ value, locked, onRate, disabled = false }: RatingControlProps) {
  return (
    <div className="rating-control">
      <div className="rating-control__buttons" role="group" aria-label="人工星级">
        {ratings.map((stars) => (
          <Button
            key={stars}
            type="button"
            size="sm"
            variant={value === stars ? 'default' : 'outline'}
            aria-label={`设为${stars}星`}
            aria-pressed={value === stars}
            disabled={disabled}
            onClick={() => void onRate(stars, true)}
          >
            {stars === 0 ? '0' : '★'.repeat(stars)}
          </Button>
        ))}
      </div>
      <span className="rating-control__lock" aria-label={locked ? '人工星级已锁定' : '人工星级未锁定'}>
        {locked ? <Lock size={14} /> : <LockOpen size={14} />}
      </span>
    </div>
  )
}
