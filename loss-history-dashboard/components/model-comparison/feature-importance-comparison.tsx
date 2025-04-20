"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts"

// Mock data for feature importance comparison
const mockFeatureImportanceComparison = {
  model1: [
    { feature: "ClaimAmount", importance: 0.22, normalized: 100 },
    { feature: "PropertyAge", importance: 0.18, normalized: 82 },
    { feature: "ClaimHistory", importance: 0.15, normalized: 68 },
    { feature: "LocationRiskScore", importance: 0.12, normalized: 55 },
    { feature: "PropertyValue", importance: 0.09, normalized: 41 },
    { feature: "CoverageLevel", importance: 0.07, normalized: 32 },
    { feature: "ClaimType", importance: 0.06, normalized: 27 },
    { feature: "WeatherEvents", importance: 0.04, normalized: 18 },
    { feature: "SecuritySystem", importance: 0.03, normalized: 14 },
    { feature: "OwnershipDuration", importance: 0.02, normalized: 9 },
  ],
  model2: [
    { feature: "ClaimAmount", importance: 0.25, normalized: 100 },
    { feature: "ClaimHistory", importance: 0.2, normalized: 80 },
    { feature: "PropertyAge", importance: 0.15, normalized: 60 },
    { feature: "LocationRiskScore", importance: 0.1, normalized: 40 },
    { feature: "ClaimType", importance: 0.08, normalized: 32 },
    { feature: "PropertyValue", importance: 0.07, normalized: 28 },
    { feature: "CoverageLevel", importance: 0.06, normalized: 24 },
    { feature: "WeatherEvents", importance: 0.04, normalized: 16 },
    { feature: "OwnershipDuration", importance: 0.03, normalized: 12 },
    { feature: "SecuritySystem", importance: 0.02, normalized: 8 },
  ],
  model3: [
    { feature: "ClaimHistory", importance: 0.24, normalized: 100 },
    { feature: "ClaimAmount", importance: 0.2, normalized: 83 },
    { feature: "LocationRiskScore", importance: 0.16, normalized: 67 },
    { feature: "PropertyAge", importance: 0.12, normalized: 50 },
    { feature: "ClaimType", importance: 0.09, normalized: 38 },
    { feature: "WeatherEvents", importance: 0.07, normalized: 29 },
    { feature: "PropertyValue", importance: 0.05, normalized: 21 },
    { feature: "CoverageLevel", importance: 0.04, normalized: 17 },
    { feature: "SecuritySystem", importance: 0.02, normalized: 8 },
    { feature: "OwnershipDuration", importance: 0.01, normalized: 4 },
  ],
}

// Prepare data for comparison chart
const prepareComparisonData = (modelData: Record<string, any[]>) => {
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
        result[modelId] = featureData ? featureData.normalized : 0
      })

      return result
    })
    .sort((a, b) => {
      // Sort by the average importance across models
      const aAvg =
        Object.entries(a)
          .filter(([key]) => key !== "feature")
          .reduce((sum, [_, value]) => sum + (value as number), 0) /
        (Object.keys(a).length - 1)

      const bAvg =
        Object.entries(b)
          .filter(([key]) => key !== "feature")
          .reduce((sum, [_, value]) => sum + (value as number), 0) /
        (Object.keys(b).length - 1)

      return bAvg - aAvg
    })
}

interface FeatureImportanceComparisonProps {
  modelIds: string[]
  modelNames?: Record<string, string>
}

export function FeatureImportanceComparison({
  modelIds,
  modelNames = {
    model1: "Loss Prediction",
    model2: "Claim Amount",
    model3: "Fraud Detection",
  },
}: FeatureImportanceComparisonProps) {
  const [loading, setLoading] = useState(true)
  const [comparisonData, setComparisonData] = useState<any[]>([])
  const [modelData, setModelData] = useState<Record<string, any[]>>({})

  useEffect(() => {
    // Simulate API call to get feature importance data for multiple models
    const fetchFeatureImportance = async () => {
      // In a real implementation, this would fetch from your API
      // const promises = modelIds.map(id => fetch(`/api/feature-importance?modelId=${id}`).then(res => res.json()))
      // const results = await Promise.all(promises)

      // Simulate loading delay
      setTimeout(() => {
        // Filter mock data to only include requested models
        const filteredData = Object.fromEntries(
          Object.entries(mockFeatureImportanceComparison).filter(([key]) => modelIds.includes(key)),
        )

        setModelData(filteredData)
        setComparisonData(prepareComparisonData(filteredData))
        setLoading(false)
      }, 1200)
    }

    if (modelIds.length > 0) {
      fetchFeatureImportance()
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
        <CardTitle>Feature Importance Comparison</CardTitle>
        <CardDescription>Compare which features matter most across different models</CardDescription>
      </CardHeader>
      <CardContent>
        {loading ? (
          <Skeleton className="h-[500px] w-full" />
        ) : (
          <div className="h-[500px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={comparisonData} layout="vertical" margin={{ top: 20, right: 30, left: 120, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} />
                <XAxis type="number" domain={[0, 100]} tickFormatter={(value) => `${value}%`} />
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
                                <span className="font-medium">{entry.value}%</span>
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
