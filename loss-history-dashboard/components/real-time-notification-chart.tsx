"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useWebSocketContext } from "@/components/websocket-provider"
import { ChartContainer } from "@/components/ui/chart"
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts"
import { cn } from "@/lib/utils"

interface ChartData {
  time: string
  critical: number
  high: number
  medium: number
  low: number
}

interface RealTimeNotificationChartProps {
  className?: string
}

export function RealTimeNotificationChart({ className }: RealTimeNotificationChartProps) {
  const [chartData, setChartData] = useState<ChartData[]>([])
  const { lastMessage } = useWebSocketContext()
  const [counters, setCounters] = useState({
    critical: 0,
    high: 0,
    medium: 0,
    low: 0,
  })

  // Initialize chart with empty data points
  useEffect(() => {
    const initialData: ChartData[] = []
    const now = new Date()

    // Create data points for the last 10 minutes with 1-minute intervals
    for (let i = 9; i >= 0; i--) {
      const time = new Date(now.getTime() - i * 60000)
      initialData.push({
        time: time.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        critical: 0,
        high: 0,
        medium: 0,
        low: 0,
      })
    }

    setChartData(initialData)
  }, [])

  // Update chart data when new notifications arrive
  useEffect(() => {
    if (lastMessage && lastMessage.type === "notification") {
      const priority = lastMessage.data.priority || "low"

      // Update counters
      setCounters((prev) => ({
        ...prev,
        [priority]: prev[priority] + 1,
      }))

      // Add new data point every minute
      const now = new Date()
      const timeStr = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })

      setChartData((prev) => {
        // Check if we already have a data point for this minute
        const existingIndex = prev.findIndex((item) => item.time === timeStr)

        if (existingIndex >= 0) {
          // Update existing data point
          const newData = [...prev]
          newData[existingIndex] = {
            ...newData[existingIndex],
            [priority]: newData[existingIndex][priority] + 1,
          }
          return newData
        } else {
          // Add new data point and remove oldest if we have more than 10
          const newData = [
            ...prev,
            {
              time: timeStr,
              critical: priority === "critical" ? 1 : 0,
              high: priority === "high" ? 1 : 0,
              medium: priority === "medium" ? 1 : 0,
              low: priority === "low" ? 1 : 0,
            },
          ]

          if (newData.length > 10) {
            return newData.slice(1)
          }

          return newData
        }
      })
    }
  }, [lastMessage])

  return (
    <Card className={cn(className)}>
      <CardHeader className="pb-2">
        <CardTitle className="text-lg">Real-Time Notification Trend</CardTitle>
      </CardHeader>
      <CardContent>
        <ChartContainer
          config={{
            critical: {
              label: "Critical",
              color: "hsl(var(--chart-purple))",
            },
            high: {
              label: "High",
              color: "hsl(var(--chart-red))",
            },
            medium: {
              label: "Medium",
              color: "hsl(var(--chart-amber))",
            },
            low: {
              label: "Low",
              color: "hsl(var(--chart-blue))",
            },
          }}
          className="h-[300px]"
        >
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart
              data={chartData}
              margin={{
                top: 10,
                right: 30,
                left: 0,
                bottom: 0,
              }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Area
                type="monotone"
                dataKey="critical"
                stackId="1"
                stroke="var(--color-critical)"
                fill="var(--color-critical)"
                fillOpacity={0.6}
              />
              <Area
                type="monotone"
                dataKey="high"
                stackId="1"
                stroke="var(--color-high)"
                fill="var(--color-high)"
                fillOpacity={0.6}
              />
              <Area
                type="monotone"
                dataKey="medium"
                stackId="1"
                stroke="var(--color-medium)"
                fill="var(--color-medium)"
                fillOpacity={0.6}
              />
              <Area
                type="monotone"
                dataKey="low"
                stackId="1"
                stroke="var(--color-low)"
                fill="var(--color-low)"
                fillOpacity={0.6}
              />
            </AreaChart>
          </ResponsiveContainer>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}
