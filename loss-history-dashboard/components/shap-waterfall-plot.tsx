"use client"

import { useState, useEffect } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  type TooltipProps,
} from "recharts"
import { ChartContainer } from "@/components/ui/chart"

// Mock SHAP waterfall data
const mockWaterfallData = [
  { feature: "Base Value", value: 0.5, cumulative: 0.5, type: "base" },
  { feature: "ClaimAmount", value: 0.15, cumulative: 0.65, type: "positive" },
  { feature: "PropertyAge", value: -0.08, cumulative: 0.57, type: "negative" },
  { feature: "ClaimHistory", value: 0.12, cumulative: 0.69, type: "positive" },
  { feature: "LocationRiskScore", value: 0.09, cumulative: 0.78, type: "positive" },
  { feature: "PropertyValue", value: -0.06, cumulative: 0.72, type: "negative" },
  { feature: "CoverageLevel", value: 0.05, cumulative: 0.77, type: "positive" },
  { feature: "ClaimType", value: -0.04, cumulative: 0.73, type: "negative" },
  { feature: "WeatherEvents", value: 0.06, cumulative: 0.79, type: "positive" },
  { feature: "SecuritySystem", value: -0.03, cumulative: 0.76, type: "negative" },
  { feature: "OwnershipDuration", value: 0.04, cumulative: 0.8, type: "positive" },
  { feature: "Final Prediction", value: 0.8, cumulative: 0.8, type: "final" },
]

// Custom tooltip for the waterfall chart
const WaterfallTooltip = ({ active, payload }: TooltipProps<number, string>) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload
    return (
      <div className="bg-background border rounded-md shadow-md p-2 text-sm">
        <p className="font-medium">{data.feature}</p>
        {data.type !== "base" && data.type !== "final" && (
          <p>
            Contribution:{" "}
            <span className={data.value > 0 ? "text-green-500" : "text-red-500"}>
              {data.value > 0 ? "+" : ""}
              {data.value.toFixed(3)}
            </span>
          </p>
        )}
        <p>Cumulative: {data.cumulative.toFixed(3)}</p>
        {data.type === "base" && <p>Starting prediction value</p>}
        {data.type === "final" && <p>Final prediction value</p>}
      </div>
    )
  }
  return null
}

interface ShapWaterfallPlotProps {
  modelId?: string
  predictionId?: string
}

export function ShapWaterfallPlot({ modelId = "model1", predictionId }: ShapWaterfallPlotProps) {
  const [loading, setLoading] = useState(true)
  const [waterfallData, setWaterfallData] = useState(mockWaterfallData)

  useEffect(() => {
    // Simulate API call to get SHAP waterfall data
    const fetchWaterfallData = async () => {
      // In a real implementation, this would fetch from your API
      // const response = await fetch(`/api/shap/waterfall?modelId=${modelId}&predictionId=${predictionId}`);
      // const data = await response.json();

      // Simulate loading delay
      setTimeout(() => {
        setWaterfallData(mockWaterfallData)
        setLoading(false)
      }, 1000)
    }

    fetchWaterfallData()
  }, [modelId, predictionId])

  return (
    <Card className="shadow-md">
      <CardContent className="pt-6">
        {loading ? (
          <Skeleton className="h-[400px] w-full" />
        ) : (
          <div className="space-y-2">
            <div className="text-sm text-muted-foreground">
              This waterfall chart shows how each feature contributes to the final prediction, starting from a base
              value and adding or subtracting each feature's impact.
            </div>
            <ChartContainer className="h-[400px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={waterfallData} margin={{ top: 20, right: 30, left: 20, bottom: 70 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="feature" angle={-45} textAnchor="end" height={70} tick={{ fontSize: 12 }} />
                  <YAxis
                    domain={[0, 1]}
                    tickFormatter={(value) => value.toFixed(1)}
                    label={{ value: "Prediction Value", angle: -90, position: "insideLeft" }}
                  />
                  <Tooltip content={<WaterfallTooltip />} />
                  <ReferenceLine y={0.5} stroke="#666" strokeDasharray="3 3" />
                  <Bar dataKey="value" fill="#8884d8" radius={[4, 4, 0, 0]}>
                    {waterfallData.map((entry, index) => {
                      let color = "#8884d8"
                      if (entry.type === "positive") color = "#10b981"
                      if (entry.type === "negative") color = "#ef4444"
                      if (entry.type === "base") color = "#6b7280"
                      if (entry.type === "final") color = "#3b82f6"

                      return <Bar key={`cell-${index}`} fill={color} />
                    })}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </ChartContainer>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
