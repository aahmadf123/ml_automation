"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from "recharts"

// Mock SHAP data for comparison
const mockShapComparison = {
  model1: [
    { feature: "ClaimAmount", impact: 0.85, direction: "positive" },
    { feature: "PropertyAge", impact: -0.72, direction: "negative" },
    { feature: "ClaimHistory", impact: 0.68, direction: "positive" },
    { feature: "LocationRiskScore", impact: 0.61, direction: "positive" },
    { feature: "PropertyValue", impact: -0.58, direction: "negative" },
    { feature: "CoverageLevel", impact: 0.52, direction: "positive" },
    { feature: "ClaimType", impact: -0.48, direction: "negative" },
    { feature: "WeatherEvents", impact: 0.45, direction: "positive" },
    { feature: "SecuritySystem", impact: -0.42, direction: "negative" },
    { feature: "OwnershipDuration", impact: 0.38, direction: "positive" },
  ],
  model2: [
    { feature: "ClaimAmount", impact: 0.92, direction: "positive" },
    { feature: "ClaimHistory", impact: 0.78, direction: "positive" },
    { feature: "PropertyAge", impact: -0.65, direction: "negative" },
    { feature: "LocationRiskScore", impact: 0.55, direction: "positive" },
    { feature: "ClaimType", impact: -0.52, direction: "negative" },
    { feature: "PropertyValue", impact: -0.48, direction: "negative" },
    { feature: "CoverageLevel", impact: 0.45, direction: "positive" },
    { feature: "WeatherEvents", impact: 0.42, direction: "positive" },
    { feature: "OwnershipDuration", impact: 0.35, direction: "positive" },
    { feature: "SecuritySystem", impact: -0.32, direction: "negative" },
  ],
  model3: [
    { feature: "ClaimHistory", impact: 0.88, direction: "positive" },
    { feature: "ClaimAmount", impact: 0.75, direction: "positive" },
    { feature: "LocationRiskScore", impact: 0.68, direction: "positive" },
    { feature: "PropertyAge", impact: -0.62, direction: "negative" },
    { feature: "ClaimType", impact: -0.58, direction: "negative" },
    { feature: "WeatherEvents", impact: 0.52, direction: "positive" },
    { feature: "PropertyValue", impact: -0.45, direction: "negative" },
    { feature: "CoverageLevel", impact: 0.38, direction: "positive" },
    { feature: "SecuritySystem", impact: -0.35, direction: "negative" },
    { feature: "OwnershipDuration", impact: 0.28, direction: "positive" },
  ],
}

// Prepare data for comparison chart
const prepareShapComparisonData = (modelData: Record<string, any[]>) => {
  const features = new Set<string>()

  // Collect all unique features
  Object.values(modelData).forEach((modelFeatures) => {
    modelFeatures.forEach((item) => features.add(item.feature))
  })

  // Create comparison data
  return Array.from(features)
    .map((feature) => {
      const result: Record<string, any> = { feature }

      Object.entries(modelData).forEach(([modelId, modelFeatures]) => {
        const featureData = modelFeatures.find((item) => item.feature === feature)
        result[modelId] = featureData ? featureData.impact : 0
      })

      return result
    })
    .sort((a, b) => {
      // Sort by the average absolute impact across models
      const aAvg = Math.abs(
        Object.entries(a)
          .filter(([key]) => key !== "feature")
          .reduce((sum, [_, value]) => sum + Math.abs(value as number), 0) /
          (Object.keys(a).length - 1),
      )

      const bAvg = Math.abs(
        Object.entries(b)
          .filter(([key]) => key !== "feature")
          .reduce((sum, [_, value]) => sum + Math.abs(value as number), 0) /
          (Object.keys(b).length - 1),
      )

      return bAvg - aAvg
    })
}

interface ShapComparisonProps {
  modelIds: string[]
  modelNames?: Record<string, string>
}

export function ShapComparison({
  modelIds,
  modelNames = {
    model1: "Loss Prediction",
    model2: "Claim Amount",
    model3: "Fraud Detection",
  },
}: ShapComparisonProps) {
  const [loading, setLoading] = useState(true)
  const [comparisonData, setComparisonData] = useState<any[]>([])
  const [modelData, setModelData] = useState<Record<string, any[]>>({})

  useEffect(() => {
    // Simulate API call to get SHAP data for multiple models
    const fetchShapData = async () => {
      // In a real implementation, this would fetch from your API
      // const promises = modelIds.map(id => fetch(`/api/shap?modelId=${id}`).then(res => res.json()))
      // const results = await Promise.all(promises)

      // Simulate loading delay
      setTimeout(() => {
        // Filter mock data to only include requested models
        const filteredData = Object.fromEntries(
          Object.entries(mockShapComparison).filter(([key]) => modelIds.includes(key)),
        )

        setModelData(filteredData)
        setComparisonData(prepareShapComparisonData(filteredData))
        setLoading(false)
      }, 1200)
    }

    if (modelIds.length > 0) {
      fetchShapData()
    }
  }, [modelIds])

  // Generate colors for the chart
  const getModelColor = (modelId: string) => {
    const colors = {
      model1: "hsl(var(--chart-1))",
      model2: "hsl(var(--chart-2))",
      model3: "hsl(var(--chart-3))",
      model4: "hsl(var(--chart-4))",
      model5: "hsl(var(--chart-5))",
    }
    return colors[modelId as keyof typeof colors] || "hsl(var(--chart-1))"
  }

  return (
    <Card className="shadow-md">
      <CardHeader className="pb-2">
        <CardTitle>SHAP Value Comparison</CardTitle>
        <CardDescription>Compare how features impact predictions across different models</CardDescription>
      </CardHeader>
      <CardContent>
        {loading ? (
          <Skeleton className="h-[500px] w-full" />
        ) : (
          <div className="h-[500px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={comparisonData} layout="vertical" margin={{ top: 20, right: 30, left: 120, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} />
                <XAxis type="number" domain={[-1, 1]} tickFormatter={(value) => value.toFixed(1)} />
                <YAxis type="category" dataKey="feature" width={110} tick={{ fontSize: 12 }} />
                <Tooltip
                  content={({ active, payload, label }) => {
                    if (active && payload && payload.length) {
                      return (
                        <div className="bg-background border rounded-md shadow-md p-3 text-sm">
                          <p className="font-medium mb-1">{label}</p>
                          <div className="space-y-1">
                            {payload.map((entry, index) => (
                              <div key={index} className="flex items-center gap-2">
                                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: entry.color }} />
                                <span>{modelNames[entry.dataKey] || entry.dataKey}:</span>
                                <span
                                  className={`font-medium ${Number(entry.value) > 0 ? "text-green-500" : "text-red-500"}`}
                                >
                                  {Number(entry.value).toFixed(2)}
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )
                    }
                    return null
                  }}
                />
                <Legend formatter={(value) => modelNames[value] || value} />
                <ReferenceLine x={0} stroke="#666" />
                {modelIds.map((modelId) => (
                  <Bar key={modelId} dataKey={modelId} fill={getModelColor(modelId)} radius={[0, 4, 4, 0]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
