"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Skeleton } from "@/components/ui/skeleton"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  type TooltipProps,
} from "recharts"
import { ChartContainer, ChartTooltip } from "@/components/ui/chart"

// Mock SHAP data
const mockShapData = {
  summary: [
    { feature: "ClaimAmount", impact: 0.85, absImpact: 0.85, direction: "positive" },
    { feature: "PropertyAge", impact: -0.72, absImpact: 0.72, direction: "negative" },
    { feature: "ClaimHistory", impact: 0.68, absImpact: 0.68, direction: "positive" },
    { feature: "LocationRiskScore", impact: 0.61, absImpact: 0.61, direction: "positive" },
    { feature: "PropertyValue", impact: -0.58, absImpact: 0.58, direction: "negative" },
    { feature: "CoverageLevel", impact: 0.52, absImpact: 0.52, direction: "positive" },
    { feature: "ClaimType", impact: -0.48, absImpact: 0.48, direction: "negative" },
    { feature: "WeatherEvents", impact: 0.45, absImpact: 0.45, direction: "positive" },
    { feature: "SecuritySystem", impact: -0.42, absImpact: 0.42, direction: "negative" },
    { feature: "OwnershipDuration", impact: 0.38, absImpact: 0.38, direction: "positive" },
    { feature: "PriorClaims", impact: 0.35, absImpact: 0.35, direction: "positive" },
    { feature: "ConstructionType", impact: -0.32, absImpact: 0.32, direction: "negative" },
    { feature: "Deductible", impact: -0.28, absImpact: 0.28, direction: "negative" },
    { feature: "InsuranceScore", impact: 0.25, absImpact: 0.25, direction: "positive" },
    { feature: "OccupancyType", impact: -0.22, absImpact: 0.22, direction: "negative" },
  ],
  waterfall: [
    { feature: "Base Value", value: 0.5, cumulative: 0.5, type: "base" },
    { feature: "ClaimAmount", value: 0.15, cumulative: 0.65, type: "positive" },
    { feature: "PropertyAge", value: -0.08, cumulative: 0.57, type: "negative" },
    { feature: "ClaimHistory", value: 0.12, cumulative: 0.69, type: "positive" },
    { feature: "LocationRiskScore", value: 0.09, cumulative: 0.78, type: "positive" },
    { feature: "PropertyValue", value: -0.06, cumulative: 0.72, type: "negative" },
    { feature: "Other Features", value: 0.08, cumulative: 0.8, type: "positive" },
    { feature: "Final Prediction", value: 0.8, cumulative: 0.8, type: "final" },
  ],
}

// Custom tooltip for the SHAP summary chart
const ShapTooltip = ({ active, payload }: TooltipProps<number, string>) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload
    return (
      <div className="bg-background border rounded-md shadow-md p-2 text-sm">
        <p className="font-medium">{data.feature}</p>
        <p>
          Impact: <span className={data.impact > 0 ? "text-green-500" : "text-red-500"}>{data.impact.toFixed(3)}</span>
        </p>
        <p>Direction: {data.direction === "positive" ? "Increases prediction" : "Decreases prediction"}</p>
      </div>
    )
  }
  return null
}

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
      </div>
    )
  }
  return null
}

interface ShapSummaryPlotProps {
  modelId?: string
  predictionId?: string
}

export function ShapSummaryPlot({ modelId = "model1", predictionId }: ShapSummaryPlotProps) {
  const [loading, setLoading] = useState(true)
  const [shapData, setShapData] = useState(mockShapData)

  useEffect(() => {
    // Simulate API call to get SHAP data
    const fetchShapData = async () => {
      // In a real implementation, this would fetch from your API
      // const response = await fetch(`/api/shap?modelId=${modelId}&predictionId=${predictionId}`);
      // const data = await response.json();

      // Simulate loading delay
      setTimeout(() => {
        setShapData(mockShapData)
        setLoading(false)
      }, 1500)
    }

    fetchShapData()
  }, [modelId, predictionId])

  // Sort data by absolute impact for the summary view
  const sortedSummaryData = [...shapData.summary].sort((a, b) => b.absImpact - a.absImpact)

  return (
    <Card className="shadow-md">
      <CardHeader className="pb-2">
        <CardTitle>SHAP Feature Impact</CardTitle>
        <CardDescription>
          {predictionId
            ? "How each feature contributed to this specific prediction"
            : "How features influence model predictions overall"}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="summary" className="space-y-4">
          <TabsList>
            <TabsTrigger value="summary">Feature Impact Summary</TabsTrigger>
            <TabsTrigger value="waterfall">Prediction Waterfall</TabsTrigger>
          </TabsList>

          <TabsContent value="summary">
            {loading ? (
              <Skeleton className="h-[400px] w-full" />
            ) : (
              <ChartContainer
                config={{
                  impact: {
                    label: "Feature Impact",
                    color: "hsl(var(--chart-1))",
                  },
                }}
                className="h-[400px]"
              >
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={sortedSummaryData}
                    layout="vertical"
                    margin={{ top: 20, right: 30, left: 120, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} />
                    <XAxis type="number" domain={[-1, 1]} tickFormatter={(value) => value.toFixed(1)} />
                    <YAxis type="category" dataKey="feature" width={110} tick={{ fontSize: 12 }} />
                    <ChartTooltip content={<ShapTooltip />} />
                    <Bar dataKey="impact" fill="var(--color-impact)" radius={[0, 4, 4, 0]}>
                      {sortedSummaryData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.impact > 0 ? "#10b981" : "#ef4444"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </ChartContainer>
            )}
          </TabsContent>

          <TabsContent value="waterfall">
            {loading ? (
              <Skeleton className="h-[400px] w-full" />
            ) : (
              <div className="h-[400px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={shapData.waterfall} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="feature" />
                    <YAxis domain={[0, 1]} tickFormatter={(value) => value.toFixed(1)} />
                    <Tooltip content={<WaterfallTooltip />} />
                    <Bar dataKey="value" fill="#8884d8">
                      {shapData.waterfall.map((entry, index) => {
                        let color = "#8884d8"
                        if (entry.type === "positive") color = "#10b981"
                        if (entry.type === "negative") color = "#ef4444"
                        if (entry.type === "base") color = "#6b7280"
                        if (entry.type === "final") color = "#3b82f6"

                        return <Cell key={`cell-${index}`} fill={color} />
                      })}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
