"use client"

import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { AlertCircle, AlertTriangle, Info } from "lucide-react"

interface SeverityFilterProps {
  onSeverityChange: (severity: string) => void
  className?: string
}

export function SeverityFilter({ onSeverityChange, className }: SeverityFilterProps) {
  return (
    <div className={className}>
      <Select onValueChange={onSeverityChange}>
        <SelectTrigger className="w-[180px]">
          <SelectValue placeholder="Filter by severity" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="all">
            <div className="flex items-center">
              <span>All Severities</span>
            </div>
          </SelectItem>
          <SelectItem value="critical">
            <div className="flex items-center">
              <AlertCircle className="mr-2 h-4 w-4 text-red-500" />
              <span>Critical</span>
              <Badge variant="outline" className="ml-auto">
                High
              </Badge>
            </div>
          </SelectItem>
          <SelectItem value="warning">
            <div className="flex items-center">
              <AlertTriangle className="mr-2 h-4 w-4 text-amber-500" />
              <span>Warning</span>
              <Badge variant="outline" className="ml-auto">
                Medium
              </Badge>
            </div>
          </SelectItem>
          <SelectItem value="info">
            <div className="flex items-center">
              <Info className="mr-2 h-4 w-4 text-blue-500" />
              <span>Info</span>
              <Badge variant="outline" className="ml-auto">
                Low
              </Badge>
            </div>
          </SelectItem>
        </SelectContent>
      </Select>
    </div>
  )
}
