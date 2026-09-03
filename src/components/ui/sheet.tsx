import * as React from 'react'
import * as DialogPrimitive from '@radix-ui/react-dialog'
import { X } from 'lucide-react'
import { cn } from '../../lib/utils'
import { DialogDescription, DialogFooter, DialogHeader, DialogOverlay, DialogTitle } from './dialog'

const Sheet = DialogPrimitive.Root
const SheetTrigger = DialogPrimitive.Trigger
const SheetClose = DialogPrimitive.Close

function SheetContent({ className, children, side = 'right', ...props }: React.ComponentProps<typeof DialogPrimitive.Content> & { side?: 'top' | 'right' | 'bottom' | 'left' }) {
  const sideClasses = {
    top: 'inset-x-0 top-0 border-b',
    right: 'inset-y-0 right-0 h-full w-[min(420px,90vw)] border-l',
    bottom: 'inset-x-0 bottom-0 border-t',
    left: 'inset-y-0 left-0 h-full w-[min(420px,90vw)] border-r',
  }
  return (
    <DialogPrimitive.Portal>
      <DialogOverlay />
      <DialogPrimitive.Content
        className={cn('fixed z-50 bg-card p-5 text-foreground shadow-lg outline-none', sideClasses[side], className)}
        {...props}
      >
        {children}
        <DialogPrimitive.Close aria-label="关闭" className="absolute right-3 top-3 rounded-md p-1 text-muted-foreground hover:bg-muted hover:text-foreground">
          <X className="size-4" />
        </DialogPrimitive.Close>
      </DialogPrimitive.Content>
    </DialogPrimitive.Portal>
  )
}

export {
  Sheet,
  SheetClose,
  SheetContent,
  DialogDescription as SheetDescription,
  DialogFooter as SheetFooter,
  DialogHeader as SheetHeader,
  DialogTitle as SheetTitle,
  SheetTrigger,
}
