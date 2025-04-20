"use client"

import { Card, CardContent } from "@/components/ui/card"
import { AlertCircle, AlertTriangle, Info } from "lucide-react"
import type { NotificationData } from "@/lib/notification-data"

interface NotificationSummaryCardsProps {
  data: NotificationData[]
}

export function NotificationSummaryCards({ data }: NotificationSummaryCardsProps) {
  // Count notifications by priority
  const criticalCount = data.filter((n) => n.priority === "critical").length
  const highCount = data.filter((n) => n.priority === "high").length
  const mediumCount = data.filter((n) => n.priority === "medium").length
  const lowCount = data.filter((n) => n.priority === "low").length

  // Count unresolved notifications by priority
  const unresolvedCritical = data.filter((n) => n.priority === "critical" && !n.resolved).length
  const unresolvedHigh = data.filter((n) => n.priority === "high" && !n.resolved).length
  const unresolvedMedium = data.filter((n) => n.priority === "medium" && !n.resolved).length
  const unresolvedLow = data.filter((n) => n.priority === "low" && !n.resolved).length

  // Calculate resolution rates
  const criticalResolutionRate =
    criticalCount > 0 ? Math.round(((criticalCount - unresolvedCritical) / criticalCount) * 100) : 0
  const highResolutionRate = highCount > 0 ? Math.round(((highCount - unresolvedHigh) / highCount) * 100) : 0
  const mediumResolutionRate = mediumCount > 0 ? Math.round(((mediumCount - unresolvedMedium) / mediumCount) * 100) : 0
  const lowResolutionRate = lowCount > 0 ? Math.round(((lowCount - unresolvedLow) / lowCount) * 100) : 0

  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
      <Card className="border-l-4 border-l-purple-500">
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-muted-foreground">Critical</p>
              <div className="flex items-baseline gap-1">
                <p className="text-2xl font-bold">{criticalCount}</p>
                <p className="text-sm text-muted-foreground">notifications</p>
              </div>
            </div>
            <div className="p-2 bg-purple-100 dark:bg-purple-900/20 rounded-full">
              <AlertCircle className="h-5 w-5 text-purple-500" />
            </div>
          </div>
          <div className="mt-2">
            <div className="flex justify-between text-xs">
              <span>Resolution Rate</span>
              <span className="font-medium">{criticalResolutionRate}%</span>
            </div>
            <div className="mt-1 h-1.5 w-full bg-purple-100 dark:bg-purple-900/20 rounded-full overflow-hidden">
              <div className="h-full bg-purple-500 rounded-full" style={{ width: `${criticalResolutionRate}%` }} />
            </div>
          </div>
          <p className="mt-2 text-xs text-muted-foreground">{unresolvedCritical} unresolved</p>
        </CardContent>
      </Card>

      <Card className="border-l-4 border-l-red-500">
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-muted-foreground">High</p>
              <div className="flex items-baseline gap-1">
                <p className="text-2xl font-bold">{highCount}</p>
                <p className="text-sm text-muted-foreground">notifications</p>
              </div>
            </div>
            <div className="p-2 bg-red-100 dark:bg-red-900/20 rounded-full">
              <AlertCircle className="h-5 w-5 text-red-500" />
            </div>
          </div>
          <div className="mt-2">
            <div className="flex justify-between text-xs">
              <span>Resolution Rate</span>
              <span className="font-medium">{highResolutionRate}%</span>
            </div>
            <div className="mt-1 h-1.5 w-full bg-red-100 dark:bg-red-900/20 rounded-full overflow-hidden">
              <div className="h-full bg-red-500 rounded-full" style={{ width: `${highResolutionRate}%` }} />
            </div>
          </div>
          <p className="mt-2 text-xs text-muted-foreground">{unresolvedHigh} unresolved</p>
        </CardContent>
      </Card>

      <Card className="border-l-4 border-l-amber-500">
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-muted-foreground">Medium</p>
              <div className="flex items-baseline gap-1">
                <p className="text-2xl font-bold">{mediumCount}</p>
                <p className="text-sm text-muted-foreground">notifications</p>
              </div>
            </div>
            <div className="p-2 bg-amber-100 dark:bg-amber-900/20 rounded-full">
              <AlertTriangle className="h-5 w-5 text-amber-500" />
            </div>
          </div>
          <div className="mt-2">
            <div className="flex justify-between text-xs">
              <span>Resolution Rate</span>
              <span className="font-medium">{mediumResolutionRate}%</span>
            </div>
            <div className="mt-1 h-1.5 w-full bg-amber-100 dark:bg-amber-900/20 rounded-full overflow-hidden">
              <div className="h-full bg-amber-500 rounded-full" style={{ width: `${mediumResolutionRate}%` }} />
            </div>
          </div>
          <p className="mt-2 text-xs text-muted-foreground">{unresolvedMedium} unresolved</p>
        </CardContent>
      </Card>

      <Card className="border-l-4 border-l-blue-500">
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-muted-foreground">Low</p>
              <div className="flex items-baseline gap-1">
                <p className="text-2xl font-bold">{lowCount}</p>
                <p className="text-sm text-muted-foreground">notifications</p>
              </div>
            </div>
            <div className="p-2 bg-blue-100 dark:bg-blue-900/20 rounded-full">
              <Info className="h-5 w-5 text-blue-500" />
            </div>
          </div>
          <div className="mt-2">
            <div className="flex justify-between text-xs">
              <span>Resolution Rate</span>
              <span className="font-medium">{lowResolutionRate}%</span>
            </div>
            <div className="mt-1 h-1.5 w-full bg-blue-100 dark:bg-blue-900/20 rounded-full overflow-hidden">
              <div className="h-full bg-blue-500 rounded-full" style={{ width: `${lowResolutionRate}%` }} />
            </div>
          </div>
          <p className="mt-2 text-xs text-muted-foreground">{unresolvedLow} unresolved</p>
        </CardContent>
      </Card>
    </div>
  )
}
