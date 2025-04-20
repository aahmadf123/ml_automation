"use client"

import { useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { ArrowRight, RefreshCw } from "lucide-react"
import { Badge } from "@/components/ui/badge"

interface ColumnMapperProps {
  sourceColumns: string[]
  targetColumns: string[]
  mappings: Record<string, string>
  onMappingChange: (mappings: Record<string, string>) => void
}

export function ColumnMapper({ sourceColumns, targetColumns, mappings, onMappingChange }: ColumnMapperProps) {
  const [localMappings, setLocalMappings] = useState<Record<string, string>>(mappings)

  const handleMappingChange = (sourceColumn: string, targetColumn: string) => {
    const newMappings = { ...localMappings, [sourceColumn]: targetColumn }
    setLocalMappings(newMappings)
    onMappingChange(newMappings)
  }

  const autoMapColumns = () => {
    const newMappings = { ...localMappings }

    // Try to auto-map columns based on name similarity
    sourceColumns.forEach((sourceCol) => {
      const normalizedSourceCol = sourceCol.toLowerCase().replace(/[_\s]/g, "")

      // Find the best match in target columns
      const bestMatch = targetColumns.find((targetCol) => {
        const normalizedTargetCol = targetCol.toLowerCase().replace(/[_\s]/g, "")
        return normalizedSourceCol === normalizedTargetCol
      })

      if (bestMatch) {
        newMappings[sourceCol] = bestMatch
      }
    })

    setLocalMappings(newMappings)
    onMappingChange(newMappings)
  }

  const clearMappings = () => {
    const emptyMappings = {} as Record<string, string>
    setLocalMappings(emptyMappings)
    onMappingChange(emptyMappings)
  }

  // Count mapped columns
  const mappedCount = Object.keys(localMappings).filter((key) => localMappings[key]).length

  return (
    <div className="space-y-4">
      <div className="flex flex-col sm:flex-row justify-between gap-4">
        <div className="flex items-center space-x-2">
          <Badge variant="outline" className="text-sm">
            {mappedCount} of {sourceColumns.length} columns mapped
          </Badge>
        </div>
        <div className="flex items-center space-x-2">
          <Button variant="outline" size="sm" onClick={clearMappings}>
            Clear All
          </Button>
          <Button variant="outline" size="sm" onClick={autoMapColumns}>
            <RefreshCw className="mr-2 h-3 w-3" />
            Auto-Map Columns
          </Button>
        </div>
      </div>

      <div className="grid gap-4">
        {sourceColumns.map((sourceColumn) => (
          <Card key={sourceColumn} className="overflow-hidden">
            <CardContent className="p-0">
              <div className="flex flex-col sm:flex-row items-stretch">
                <div className="bg-muted/50 p-4 sm:w-1/2 flex items-center">
                  <div>
                    <p className="font-medium">{sourceColumn}</p>
                    <p className="text-sm text-muted-foreground">Source Column</p>
                  </div>
                </div>
                <div className="hidden sm:flex items-center justify-center w-12">
                  <ArrowRight className="h-4 w-4 text-muted-foreground" />
                </div>
                <div className="p-4 sm:w-1/2 flex items-center">
                  <Select
                    value={localMappings[sourceColumn] || ""}
                    onValueChange={(value) => handleMappingChange(sourceColumn, value)}
                  >
                    <SelectTrigger className="w-full">
                      <SelectValue placeholder="Select target field" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="none">-- Not Mapped --</SelectItem>
                      {targetColumns.map((targetColumn) => (
                        <SelectItem key={targetColumn} value={targetColumn}>
                          {targetColumn}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}
