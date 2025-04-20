"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Calendar } from "@/components/ui/calendar"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { CalendarIcon, RefreshCw } from "lucide-react"
import { format } from "date-fns"
import { cn } from "@/lib/utils"

interface NotificationDashboardFiltersProps {
  onDateRangeChange: (range: { from: Date; to: Date }) => void
  onRefresh: () => void
}

export function NotificationDashboardFilters({ onDateRangeChange, onRefresh }: NotificationDashboardFiltersProps) {
  const [date, setDate] = useState<{
    from: Date
    to: Date
  }>({
    from: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), // 30 days ago
    to: new Date(),
  })

  const handleDateSelect = (selectedDate: Date | undefined) => {
    if (!selectedDate) return

    const range = {
      from: date.from,
      to: date.to,
    }

    if (!date.from || date.to) {
      range.from = selectedDate
      range.to = selectedDate
    } else if (selectedDate < date.from) {
      range.from = selectedDate
    } else {
      range.to = selectedDate
    }

    setDate(range)
    onDateRangeChange(range)
  }

  // Predefined date ranges
  const selectLast7Days = () => {
    const range = {
      from: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
      to: new Date(),
    }
    setDate(range)
    onDateRangeChange(range)
  }

  const selectLast30Days = () => {
    const range = {
      from: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
      to: new Date(),
    }
    setDate(range)
    onDateRangeChange(range)
  }

  const selectLast90Days = () => {
    const range = {
      from: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000),
      to: new Date(),
    }
    setDate(range)
    onDateRangeChange(range)
  }

  return (
    <div className="flex flex-wrap items-center gap-2 mb-6">
      <div className="flex flex-wrap items-center gap-2">
        <Button variant="outline" size="sm" onClick={selectLast7Days}>
          Last 7 days
        </Button>
        <Button variant="outline" size="sm" onClick={selectLast30Days}>
          Last 30 days
        </Button>
        <Button variant="outline" size="sm" onClick={selectLast90Days}>
          Last 90 days
        </Button>

        <Popover>
          <PopoverTrigger asChild>
            <Button
              variant="outline"
              size="sm"
              className={cn("justify-start text-left font-normal", !date && "text-muted-foreground")}
            >
              <CalendarIcon className="mr-2 h-4 w-4" />
              {date?.from ? (
                date.to ? (
                  <>
                    {format(date.from, "LLL dd, y")} - {format(date.to, "LLL dd, y")}
                  </>
                ) : (
                  format(date.from, "LLL dd, y")
                )
              ) : (
                <span>Pick a date</span>
              )}
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-auto p-0" align="start">
            <Calendar
              initialFocus
              mode="range"
              defaultMonth={date?.from}
              selected={date}
              onSelect={(range) => {
                if (range?.from && range?.to) {
                  setDate(range)
                  onDateRangeChange(range)
                }
              }}
              numberOfMonths={2}
            />
          </PopoverContent>
        </Popover>
      </div>

      <div className="ml-auto">
        <Button variant="outline" size="sm" onClick={onRefresh}>
          <RefreshCw className="mr-2 h-4 w-4" />
          Refresh
        </Button>
      </div>
    </div>
  )
}
