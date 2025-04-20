"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Calendar } from "@/components/ui/calendar"
import { Checkbox } from "@/components/ui/checkbox"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Sheet,
  SheetClose,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet"
import { CalendarIcon, Check, ChevronDown, Filter, Save, Search, SlidersHorizontal, X } from "lucide-react"
import { format } from "date-fns"
import { cn } from "@/lib/utils"

export interface FilterOption {
  id: string
  label: string
  type: "select" | "multiselect" | "date" | "daterange" | "search" | "checkbox" | "radio"
  options?: { value: string; label: string }[]
  value?: any
}

interface FilterPreset {
  id: string
  name: string
  filters: Record<string, any>
}

interface DashboardFilterProps {
  title?: string
  description?: string
  filterOptions: FilterOption[]
  onFilterChange: (filters: Record<string, any>) => void
  allowSavedFilters?: boolean
  className?: string
}

export function DashboardFilter({
  title = "Filter Data",
  description = "Apply filters to customize your view",
  filterOptions,
  onFilterChange,
  allowSavedFilters = true,
  className,
}: DashboardFilterProps) {
  const [filters, setFilters] = useState<Record<string, any>>({})
  const [activeFiltersCount, setActiveFiltersCount] = useState(0)
  const [savedFilters, setSavedFilters] = useState<FilterPreset[]>([
    {
      id: "recent",
      name: "Recent Items",
      filters: { date: { from: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000), to: new Date() } },
    },
    {
      id: "critical",
      name: "Critical Issues",
      filters: { severity: ["critical"], status: ["failed", "warning"] },
    },
  ])
  const [newPresetName, setNewPresetName] = useState("")

  const handleFilterChange = (id: string, value: any) => {
    const newFilters = { ...filters, [id]: value }

    // Remove empty filters
    if (value === "" || value === null || (Array.isArray(value) && value.length === 0)) {
      delete newFilters[id]
    }

    setFilters(newFilters)

    // Count active filters
    const count = Object.keys(newFilters).length
    setActiveFiltersCount(count)

    // Notify parent component
    onFilterChange(newFilters)
  }

  const clearFilters = () => {
    setFilters({})
    setActiveFiltersCount(0)
    onFilterChange({})
  }

  const saveCurrentFilters = () => {
    if (!newPresetName.trim()) return

    const newPreset: FilterPreset = {
      id: `preset-${Date.now()}`,
      name: newPresetName,
      filters: { ...filters },
    }

    setSavedFilters([...savedFilters, newPreset])
    setNewPresetName("")
  }

  const loadSavedFilter = (preset: FilterPreset) => {
    setFilters(preset.filters)
    setActiveFiltersCount(Object.keys(preset.filters).length)
    onFilterChange(preset.filters)
  }

  const renderFilterControl = (option: FilterOption) => {
    switch (option.type) {
      case "select":
        return (
          <Select value={filters[option.id] || ""} onValueChange={(value) => handleFilterChange(option.id, value)}>
            <SelectTrigger className="w-full">
              <SelectValue placeholder={`Select ${option.label}`} />
            </SelectTrigger>
            <SelectContent>
              <SelectGroup>
                <SelectLabel>{option.label}</SelectLabel>
                {option.options?.map((opt) => (
                  <SelectItem key={opt.value} value={opt.value}>
                    {opt.label}
                  </SelectItem>
                ))}
              </SelectGroup>
            </SelectContent>
          </Select>
        )

      case "multiselect":
        return (
          <div className="space-y-2">
            {option.options?.map((opt) => (
              <div key={opt.value} className="flex items-center space-x-2">
                <Checkbox
                  id={`${option.id}-${opt.value}`}
                  checked={(filters[option.id] || []).includes(opt.value)}
                  onCheckedChange={(checked) => {
                    const currentValues = filters[option.id] || []
                    const newValues = checked
                      ? [...currentValues, opt.value]
                      : currentValues.filter((v: string) => v !== opt.value)
                    handleFilterChange(option.id, newValues)
                  }}
                />
                <Label htmlFor={`${option.id}-${opt.value}`} className="text-sm font-normal">
                  {opt.label}
                </Label>
              </div>
            ))}
          </div>
        )

      case "daterange":
        return (
          <div className="grid gap-2">
            <Popover>
              <PopoverTrigger asChild>
                <Button
                  variant="outline"
                  className={cn(
                    "w-full justify-start text-left font-normal",
                    !filters[option.id] && "text-muted-foreground",
                  )}
                >
                  <CalendarIcon className="mr-2 h-4 w-4" />
                  {filters[option.id]?.from ? (
                    filters[option.id]?.to ? (
                      <>
                        {format(filters[option.id].from, "PPP")} - {format(filters[option.id].to, "PPP")}
                      </>
                    ) : (
                      format(filters[option.id].from, "PPP")
                    )
                  ) : (
                    <span>Pick a date range</span>
                  )}
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-auto p-0" align="start">
                <Calendar
                  initialFocus
                  mode="range"
                  defaultMonth={filters[option.id]?.from}
                  selected={filters[option.id]}
                  onSelect={(range) => handleFilterChange(option.id, range)}
                  numberOfMonths={2}
                />
              </PopoverContent>
            </Popover>
          </div>
        )

      case "search":
        return (
          <div className="relative">
            <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder={`Search ${option.label}`}
              value={filters[option.id] || ""}
              onChange={(e) => handleFilterChange(option.id, e.target.value)}
              className="pl-8"
            />
          </div>
        )

      default:
        return null
    }
  }

  return (
    <div className={cn("flex flex-wrap items-center gap-2", className)}>
      <Sheet>
        <SheetTrigger asChild>
          <Button variant="outline" size="sm" className="h-8">
            <Filter className="mr-2 h-3.5 w-3.5" />
            Filters
            {activeFiltersCount > 0 && (
              <Badge className="ml-1 h-5 w-5 rounded-full p-0 text-xs">{activeFiltersCount}</Badge>
            )}
          </Button>
        </SheetTrigger>
        <SheetContent side="right" className="w-full sm:max-w-md">
          <SheetHeader>
            <SheetTitle>{title}</SheetTitle>
            <SheetDescription>{description}</SheetDescription>
          </SheetHeader>

          <div className="mt-6 space-y-6">
            {filterOptions.map((option) => (
              <div key={option.id} className="space-y-2">
                <Label htmlFor={option.id} className="text-sm font-medium">
                  {option.label}
                </Label>
                {renderFilterControl(option)}
              </div>
            ))}
          </div>

          {allowSavedFilters && (
            <div className="mt-8 border-t pt-4">
              <h4 className="mb-2 text-sm font-medium">Saved Filters</h4>
              <div className="mb-4 space-y-2">
                {savedFilters.map((preset) => (
                  <Button
                    key={preset.id}
                    variant="outline"
                    size="sm"
                    className="mr-2 h-8"
                    onClick={() => loadSavedFilter(preset)}
                  >
                    {preset.name}
                  </Button>
                ))}
              </div>

              <div className="flex items-center space-x-2">
                <Input
                  placeholder="New filter preset name"
                  value={newPresetName}
                  onChange={(e) => setNewPresetName(e.target.value)}
                  className="h-8"
                />
                <Button size="sm" className="h-8" onClick={saveCurrentFilters} disabled={!newPresetName.trim()}>
                  <Save className="mr-2 h-3.5 w-3.5" />
                  Save
                </Button>
              </div>
            </div>
          )}

          <SheetFooter className="mt-6 flex-row justify-between">
            <Button variant="outline" size="sm" onClick={clearFilters}>
              <X className="mr-2 h-3.5 w-3.5" />
              Clear All
            </Button>
            <SheetClose asChild>
              <Button size="sm">
                <Check className="mr-2 h-3.5 w-3.5" />
                Apply Filters
              </Button>
            </SheetClose>
          </SheetFooter>
        </SheetContent>
      </Sheet>

      {/* Quick filters */}
      <Select onValueChange={(value) => loadSavedFilter(savedFilters.find((f) => f.id === value)!)}>
        <SelectTrigger className="h-8 w-[180px]">
          <SelectValue placeholder="Quick filters" />
        </SelectTrigger>
        <SelectContent>
          {savedFilters.map((preset) => (
            <SelectItem key={preset.id} value={preset.id}>
              {preset.name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      <Popover>
        <PopoverTrigger asChild>
          <Button variant="outline" size="sm" className="h-8">
            <CalendarIcon className="mr-2 h-3.5 w-3.5" />
            <span>Date Range</span>
            <ChevronDown className="ml-2 h-3.5 w-3.5" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-auto p-0" align="start">
          <Calendar
            initialFocus
            mode="range"
            defaultMonth={new Date()}
            selected={filters.date}
            onSelect={(range) => handleFilterChange("date", range)}
            numberOfMonths={2}
          />
        </PopoverContent>
      </Popover>

      <Button variant="outline" size="sm" className="h-8" onClick={clearFilters}>
        <SlidersHorizontal className="mr-2 h-3.5 w-3.5" />
        Advanced
      </Button>

      {activeFiltersCount > 0 && (
        <Button variant="ghost" size="sm" className="h-8" onClick={clearFilters}>
          <X className="mr-2 h-3.5 w-3.5" />
          Clear All
        </Button>
      )}
    </div>
  )
}
