"use client"

import { LineChart as RechartsLineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"

interface Metric {
  timestamp: number
  value: number
}

interface LineChartProps {
  data: Metric[]
  color?: string
}

export function LineChart({ data, color = "#2563eb" }: LineChartProps) {
  const formattedData = data.map((metric) => ({
    time: new Date(metric.timestamp).toLocaleTimeString(),
    value: metric.value
  }))

  return (
    <div className="h-[200px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <RechartsLineChart data={formattedData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="time" />
          <YAxis />
          <Tooltip />
          <Line
            type="monotone"
            dataKey="value"
            stroke={color}
            strokeWidth={2}
            dot={false}
          />
        </RechartsLineChart>
      </ResponsiveContainer>
    </div>
  )
} 