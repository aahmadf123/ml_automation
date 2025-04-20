"use client"

import { useState, useEffect } from "react"
import { Card } from "@/components/ui/card"
import { LineChart } from "@/components/ui/line-chart"
import { useWebSocket } from "@/lib/websocket"

interface MetricData {
  value: number
  timestamp: number
}

interface RealTimeMetricsProps {
  title: string
  description?: string
  metricKey: string
  unit?: string
  color?: string
  maxDataPoints?: number
}

export function RealTimeMetrics({
  title,
  description,
  metricKey,
  unit = "",
  color = "blue",
  maxDataPoints = 50,
}: RealTimeMetricsProps) {
  const [data, setData] = useState<MetricData[]>([])
  const { lastMessage, isConnected } = useWebSocket()

  useEffect(() => {
    if (lastMessage && lastMessage.type === "metrics_update") {
      const metricValue = lastMessage.data[metricKey]
      if (metricValue !== undefined) {
        setData((prevData) => {
          const newData = [
            ...prevData,
            { value: metricValue, timestamp: lastMessage.timestamp },
          ]
          if (newData.length > maxDataPoints) {
            return newData.slice(-maxDataPoints)
          }
          return newData
        })
      }
    }
  }, [lastMessage, metricKey, maxDataPoints])

  const currentValue = data[data.length - 1]?.value ?? 0
  const previousValue = data[data.length - 2]?.value ?? 0
  const percentChange = previousValue
    ? ((currentValue - previousValue) / previousValue) * 100
    : 0

  return (
    <Card className="p-4">
      <div className="flex flex-col space-y-2">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold">{title}</h3>
            {description && (
              <p className="text-sm text-gray-500">{description}</p>
            )}
          </div>
          <div className="text-right">
            <div className="text-2xl font-bold">
              {currentValue.toFixed(2)}
              {unit}
            </div>
            <div
              className={`text-sm ${
                percentChange >= 0 ? "text-green-500" : "text-red-500"
              }`}
            >
              {percentChange >= 0 ? "↑" : "↓"} {Math.abs(percentChange).toFixed(2)}%
            </div>
          </div>
        </div>
        <div className="h-32">
          <LineChart
            data={data.map((d) => ({
              timestamp: new Date(d.timestamp).toLocaleTimeString(),
              value: d.value,
            }))}
            color={color}
          />
        </div>
        <div className="text-xs text-gray-500">
          {isConnected ? "Connected" : "Disconnected"}
        </div>
      </div>
    </Card>
  )
} 