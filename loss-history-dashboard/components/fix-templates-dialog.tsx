"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { FixTemplatesLibrary, type FixTemplate } from "@/components/fix-templates-library"
import { FileCode } from "lucide-react"

interface FixTemplatesDialogProps {
  onApplyTemplate: (template: FixTemplate) => void
}

export function FixTemplatesDialog({ onApplyTemplate }: FixTemplatesDialogProps) {
  const [open, setOpen] = useState(false)

  const handleApplyTemplate = (template: FixTemplate) => {
    onApplyTemplate(template)
    setOpen(false)
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="icon" title="Browse fix templates">
          <FileCode className="h-4 w-4" />
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-[90vw] w-[1200px] h-[80vh] max-h-[800px] p-0">
        <DialogHeader className="p-6 pb-0">
          <DialogTitle>Fix Templates Library</DialogTitle>
          <DialogDescription>Browse and apply predefined fixes for common issues</DialogDescription>
        </DialogHeader>
        <div className="p-6 pt-2 h-full">
          <FixTemplatesLibrary onApplyTemplate={handleApplyTemplate} onClose={() => setOpen(false)} />
        </div>
      </DialogContent>
    </Dialog>
  )
}
