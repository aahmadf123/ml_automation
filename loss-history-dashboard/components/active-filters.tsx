"use client"

import { X } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { format } from "date-fns"

interface ActiveFiltersProps {
  filters: Record<string, any>
  filterLabels: Record<string, string>
  onRemoveFilter: (key: string) => void
  onClearAll: () => void
  className?: string
}

export function ActiveFilters({ filters, filterLabels, onRemoveFilter, onClearAll, className }: ActiveFiltersProps) {
  if (Object.keys(filters).length === 0) {
    return null
  }

  const formatFilterValue = (key: string, value: any) => {
    if (key === "date" && value.from) {
      if (value.to) {
        return `${format(value.from, "MMM d, yyyy")} - ${format(value.to, "MMM d, yyyy")}`
      }
      return `From ${format(value.from, "MMM d, yyyy")}`
    }

    if (Array.isArray(value)) {
      return value.join(", ")
    }

    return value
  }

  return (
    <div className={className}>
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm text-muted-foreground">Active filters:</span>

        {Object.entries(filters).map(([key, value]) => (
          <Badge key={key} variant="outline" className="flex items-center gap-1">
            <span className="font-medium">{filterLabels[key] || key}:</span>
            <span>{formatFilterValue(key, value)}</span>
            <Button
              variant="ghost"
              size="icon"
              className="h-4 w-4 rounded-full p-0"
              onClick={() => onRemoveFilter(key)}
            >
              <X className="h-3 w-3" />
              <span className="sr-only">Remove filter</span>
            </Button>
          </Badge>
        ))}

        <Button variant="ghost" size="sm" className="h-7 text-xs" onClick={onClearAll}>
          Clear all
        </Button>
      </div>
    </div>
  )
}
