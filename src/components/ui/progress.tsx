import * as React from 'react'
import * as ProgressPrimitive from '@radix-ui/react-progress'
import { cn } from '../../lib/utils'

function Progress({ className, value = 0, ...props }: React.ComponentProps<typeof ProgressPrimitive.Root>) {
  const normalized = Math.min(100, Math.max(0, value ?? 0))
  return (
    <ProgressPrimitive.Root
      className={cn('relative h-2 w-full overflow-hidden rounded-md bg-muted', className)}
      value={normalized}
      {...props}
    >
      <ProgressPrimitive.Indicator
        className="h-full w-full bg-primary transition-transform"
        style={{ transform: `translateX(-${100 - normalized}%)` }}
      />
    </ProgressPrimitive.Root>
  )
}

export { Progress }
