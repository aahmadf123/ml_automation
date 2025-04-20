"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useWebSocketContext } from "@/components/websocket-provider"
import { ArrowUpIcon } from "lucide-react"
import { cn } from "@/lib/utils"

interface MetricData {
  total: number
  critical: number
  high: number
  medium: number
  low: number
  change: {
    total: number
    critical: number
    high: number
    medium: number
    low: number
  }
}

export function RealTimeNotificationMetrics() {
  const [metrics, setMetrics] = useState<MetricData>({
    total: 0,
    critical: 0,
    high: 0,
    medium: 0,
    low: 0,
    change: {
      total: 0,
      critical: 0,
      high: 0,
      medium: 0,
      low: 0,
    },
  })

  const { lastMessage, messageHistory } = useWebSocketContext()

  // Process incoming WebSocket messages
  useEffect(() => {
    if (lastMessage && lastMessage.type === "notification") {
      const priority = lastMessage.data.priority || "low"

      setMetrics((prev) => {
        const newMetrics = { ...prev }
        newMetrics.total += 1
        newMetrics[priority] += 1

        // Track changes for animation
        newMetrics.change.total += 1
        newMetrics.change[priority] += 1

        return newMetrics
      })

      // Reset change indicators after 3 seconds
      setTimeout(() => {
        setMetrics((prev) => ({
          ...prev,
          change: {
            total: 0,
            critical: 0,
            high: 0,
            medium: 0,
            low: 0,
          },
        }))
      }, 3000)
    }
  }, [lastMessage])

  // Initialize with any existing notification messages from history
  useEffect(() => {
    const notificationMessages = messageHistory.filter((msg) => msg.type === "notification")

    if (notificationMessages.length > 0) {
      const initialMetrics = {
        total: notificationMessages.length,
        critical: 0,
        high: 0,
        medium: 0,
        low: 0,
        change: {
          total: 0,
          critical: 0,
          high: 0,
          medium: 0,
          low: 0,
        },
      }

      notificationMessages.forEach((msg) => {
        const priority = msg.data.priority || "low"
        initialMetrics[priority] += 1
      })

      setMetrics(initialMetrics)
    }
  }, [messageHistory])

  const MetricItem = ({
    label,
    value,
    change,
    color,
  }: { label: string; value: number; change: number; color: string }) => (
    <div className="flex flex-col">
      <div className="text-sm font-medium text-gray-500 dark:text-gray-400">{label}</div>
      <div className="flex items-center gap-2">
        <div className={cn("text-2xl font-bold", color)}>{value}</div>
        {change > 0 && (
          <div className="flex items-center text-green-500 text-xs font-medium">
            <ArrowUpIcon className="h-3 w-3 mr-1" />+{change}
          </div>
        )}
      </div>
    </div>
  )

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-lg">Real-Time Metrics</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <MetricItem
            label="Total"
            value={metrics.total}
            change={metrics.change.total}
            color="text-gray-900 dark:text-gray-100"
          />
          <MetricItem
            label="Critical"
            value={metrics.critical}
            change={metrics.change.critical}
            color="text-purple-600 dark:text-purple-400"
          />
          <MetricItem
            label="High"
            value={metrics.high}
            change={metrics.change.high}
            color="text-red-600 dark:text-red-400"
          />
          <MetricItem
            label="Medium"
            value={metrics.medium}
            change={metrics.change.medium}
            color="text-amber-600 dark:text-amber-400"
          />
          <MetricItem
            label="Low"
            value={metrics.low}
            change={metrics.change.low}
            color="text-blue-600 dark:text-blue-400"
          />
        </div>
      </CardContent>
    </Card>
  )
}
