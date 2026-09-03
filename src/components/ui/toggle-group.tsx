import * as React from 'react'
import * as ToggleGroupPrimitive from '@radix-ui/react-toggle-group'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '../../lib/utils'

const toggleVariants = cva(
  'inline-flex items-center justify-center rounded-md text-sm font-medium outline-none transition-colors hover:bg-muted focus-visible:ring-2 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50 data-[state=on]:bg-primary data-[state=on]:text-primary-foreground',
  {
    variants: {
      size: {
        sm: 'h-8 px-2.5',
        default: 'h-9 px-3',
        lg: 'h-10 px-4',
      },
    },
    defaultVariants: { size: 'default' },
  },
)

const ToggleGroupContext = React.createContext<VariantProps<typeof toggleVariants>>({ size: 'default' })

function ToggleGroup({ className, size, children, ...props }: React.ComponentProps<typeof ToggleGroupPrimitive.Root> & VariantProps<typeof toggleVariants>) {
  return (
    <ToggleGroupPrimitive.Root className={cn('inline-flex items-center gap-1 rounded-md border border-border p-1', className)} {...props}>
      <ToggleGroupContext.Provider value={{ size }}>{children}</ToggleGroupContext.Provider>
    </ToggleGroupPrimitive.Root>
  )
}

function ToggleGroupItem({ className, size, ...props }: React.ComponentProps<typeof ToggleGroupPrimitive.Item> & VariantProps<typeof toggleVariants>) {
  const context = React.useContext(ToggleGroupContext)
  return <ToggleGroupPrimitive.Item className={cn(toggleVariants({ size: size ?? context.size }), className)} {...props} />
}

export { ToggleGroup, ToggleGroupItem }
