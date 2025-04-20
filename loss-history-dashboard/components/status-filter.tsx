"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { CheckCircle, XCircle, AlertTriangle, Clock, Filter } from "lucide-react"
import { cn } from "@/lib/utils"

interface StatusOption {
  id: string
  label: string
  icon: React.ReactNode
  color: string
}

interface StatusFilterProps {
  onStatusChange: (statuses: string[]) => void
  className?: string
}

export function StatusFilter({ onStatusChange, className }: StatusFilterProps) {
  const [selectedStatuses, setSelectedStatuses] = useState<string[]>([])

  const statuses: StatusOption[] = [
    {
      id: "success",
      label: "Success",
      icon: <CheckCircle className="h-4 w-4" />,
      color: "text-green-500 border-green-500",
    },
    {
      id: "failed",
      label: "Failed",
      icon: <XCircle className="h-4 w-4" />,
      color: "text-red-500 border-red-500",
    },
    {
      id: "warning",
      label: "Warning",
      icon: <AlertTriangle className="h-4 w-4" />,
      color: "text-amber-500 border-amber-500",
    },
    {
      id: "running",
      label: "Running",
      icon: <Clock className="h-4 w-4" />,
      color: "text-blue-500 border-blue-500",
    },
  ]

  const toggleStatus = (statusId: string) => {
    let newSelectedStatuses: string[]

    if (selectedStatuses.includes(statusId)) {
      newSelectedStatuses = selectedStatuses.filter((id) => id !== statusId)
    } else {
      newSelectedStatuses = [...selectedStatuses, statusId]
    }

    setSelectedStatuses(newSelectedStatuses)
    onStatusChange(newSelectedStatuses)
  }

  return (
    <div className={cn("flex flex-wrap items-center gap-2", className)}>
      <div className="flex items-center mr-2">
        <Filter className="h-4 w-4 mr-1 text-muted-foreground" />
        <span className="text-sm text-muted-foreground">Status:</span>
      </div>

      {statuses.map((status) => (
        <Button
          key={status.id}
          variant={selectedStatuses.includes(status.id) ? "default" : "outline"}
          size="sm"
          className={cn("h-8", selectedStatuses.includes(status.id) ? "" : status.color)}
          onClick={() => toggleStatus(status.id)}
        >
          {status.icon}
          <span className="ml-1">{status.label}</span>
        </Button>
      ))}

      {selectedStatuses.length > 0 && (
        <Button
          variant="ghost"
          size="sm"
          className="h-8"
          onClick={() => {
            setSelectedStatuses([])
            onStatusChange([])
          }}
        >
          Clear
        </Button>
      )}
    </div>
  )
}
