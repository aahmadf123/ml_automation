"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { type NotificationData, getAverageResolutionTimeByPriority } from "@/lib/notification-data"
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, Cell } from "recharts"

interface ResolutionTimeChartProps {
  data: NotificationData[]
}

export function ResolutionTimeChart({ data }: ResolutionTimeChartProps) {
  // Get average resolution times
  const resolutionTimes = getAverageResolutionTimeByPriority(data)

  // Format data for chart
  const chartData = [
    { name: "Critical", value: resolutionTimes.critical, color: "#9333ea" },
    { name: "High", value: resolutionTimes.high, color: "#ef4444" },
    { name: "Medium", value: resolutionTimes.medium, color: "#f59e0b" },
    { name: "Low", value: resolutionTimes.low, color: "#3b82f6" },
  ]

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const minutes = payload[0].value
      let timeDisplay = ""

      if (minutes < 60) {
        timeDisplay = `${minutes} minutes`
      } else if (minutes < 1440) {
        const hours = Math.floor(minutes / 60)
        const remainingMinutes = minutes % 60
        timeDisplay = `${hours} hour${hours !== 1 ? "s" : ""}`
        if (remainingMinutes > 0) {
          timeDisplay += ` ${remainingMinutes} min`
        }
      } else {
        const days = Math.floor(minutes / 1440)
        const hours = Math.floor((minutes % 1440) / 60)
        timeDisplay = `${days} day${days !== 1 ? "s" : ""}`
        if (hours > 0) {
          timeDisplay += ` ${hours} hr`
        }
      }

      return (
        <div className="bg-background border rounded-md shadow-md p-2 text-sm">
          <p className="font-medium">{payload[0].name}</p>
          <p>Average Resolution Time: {timeDisplay}</p>
        </div>
      )
    }
    return null
  }

  // Format minutes for display
  const formatMinutes = (minutes: number) => {
    if (minutes < 60) return `${minutes}m`
    if (minutes < 1440) return `${Math.floor(minutes / 60)}h`
    return `${Math.floor(minutes / 1440)}d`
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Average Resolution Time</CardTitle>
      </CardHeader>
      <CardContent className="h-[350px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 10 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis tickFormatter={formatMinutes} />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Bar dataKey="value" name="Resolution Time" radius={[4, 4, 0, 0]}>
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}
