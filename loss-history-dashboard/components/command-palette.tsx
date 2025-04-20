"use client"

import type React from "react"

import { useState, useCallback } from "react"
import { Command as CommandPrimitive } from "cmdk"
import { useRouter } from "next/navigation"

import { cn } from "@/lib/utils"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"

const K_KEY = "/"

interface CommandPaletteProps extends React.HTMLAttributes<HTMLDivElement> {}

function CommandPalette({ className, ...props }: CommandPaletteProps) {
  const [open, setOpen] = useState(false)
  const router = useRouter()

  const onSelect = useCallback(
    (href: string) => {
      setOpen(false)
      router.push(href)
    },
    [router],
  )

  const handleOpen = useCallback(() => {
    setOpen(true)
  }, [])

  const handleClose = useCallback(() => {
    setOpen(false)
  }, [])

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <kbd
          className={cn(
            "pointer-events-none absolute right-6 top-4 rounded border bg-muted px-2 py-1.5 font-mono text-[.8em] font-semibold opacity-60",
            className,
          )}
          {...props}
        >
          {K_KEY}
        </kbd>
      </DialogTrigger>
      <DialogContent className="overflow-hidden p-0 shadow-lg">
        <DialogHeader className="pt-8 pb-6">
          <DialogTitle>Type a command or search...</DialogTitle>
        </DialogHeader>
        <CommandPrimitive.Root className="overflow-hidden rounded-md border">
          <CommandPrimitive.Input
            className="h-11 w-full rounded-md border-0 bg-transparent py-3 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-0 disabled:cursor-not-allowed disabled:opacity-50"
            placeholder="Search all commands..."
          />
          <CommandPrimitive.List className="max-h-[300px] overflow-y-auto">
            <CommandPrimitive.Empty>No results found.</CommandPrimitive.Empty>
            <CommandPrimitive.Group heading="Navigation">
              <CommandPrimitive.Item onSelect={() => onSelect("/")}>Home</CommandPrimitive.Item>
              <CommandPrimitive.Item onSelect={() => onSelect("/drift-monitor")}>Drift Monitor</CommandPrimitive.Item>
              <CommandPrimitive.Item onSelect={() => onSelect("/model-metrics")}>Model Metrics</CommandPrimitive.Item>
              <CommandPrimitive.Item onSelect={() => onSelect("/data-ingestion")}>Data Ingestion</CommandPrimitive.Item>
              <CommandPrimitive.Item onSelect={() => onSelect("/visualizations")}>Visualizations</CommandPrimitive.Item>
              <CommandPrimitive.Item onSelect={() => onSelect("/data-quality")}>Data Quality</CommandPrimitive.Item>
              <CommandPrimitive.Item onSelect={() => onSelect("/incidents")}>Incidents</CommandPrimitive.Item>
              <CommandPrimitive.Item onSelect={() => onSelect("/code-interpreter")}>
                Code Interpreter
              </CommandPrimitive.Item>
              <CommandPrimitive.Item onSelect={() => onSelect("/fix-templates")}>Fix Templates</CommandPrimitive.Item>
              <CommandPrimitive.Item onSelect={() => onSelect("/notification-analytics")}>
                Notification Analytics
              </CommandPrimitive.Item>
              <CommandPrimitive.Item onSelect={() => onSelect("/settings")}>Settings</CommandPrimitive.Item>
            </CommandPrimitive.Group>
          </CommandPrimitive.List>
        </CommandPrimitive.Root>
      </DialogContent>
    </Dialog>
  )
}

export { CommandPalette }
