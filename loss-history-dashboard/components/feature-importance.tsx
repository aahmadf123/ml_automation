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
  PieChart,
  Pie,
  Cell,
  Legend,
} from "recharts"
import { ChartContainer, ChartTooltip } from "@/components/ui/chart"

// Mock feature importance data
const mockFeatureImportanceData = {
  global: [
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
    { feature: "Other Features", importance: 0.02, normalized: 9 },
  ],
  categories: [
    { name: "Claim Characteristics", value: 0.35 },
    { name: "Property Attributes", value: 0.28 },
    { name: "Location Factors", value: 0.18 },
    { name: "Owner Profile", value: 0.12 },
    { name: "Policy Details", value: 0.07 },
  ],
}

// Colors for the pie chart
const COLORS = ["#0088FE", "#00C49F", "#FFBB28", "#FF8042", "#8884D8"]

interface FeatureImportanceProps {
  modelId?: string
}

export function FeatureImportance({ modelId = "model1" }: FeatureImportanceProps) {
  const [loading, setLoading] = useState(true)
  const [featureData, setFeatureData] = useState(mockFeatureImportanceData)

  useEffect(() => {
    // Simulate API call to get feature importance data
    const fetchFeatureImportance = async () => {
      // In a real implementation, this would fetch from your API
      // const response = await fetch(`/api/feature-importance?modelId=${modelId}`);
      // const data = await response.json();

      // Simulate loading delay
      setTimeout(() => {
        setFeatureData(mockFeatureImportanceData)
        setLoading(false)
      }, 1200)
    }

    fetchFeatureImportance()
  }, [modelId])

  // Format percentage for the pie chart
  const formatPercent = (value: number) => `${(value * 100).toFixed(0)}%`

  return (
    <Card className="shadow-md">
      <CardHeader className="pb-2">
        <CardTitle>Feature Importance</CardTitle>
        <CardDescription>Which features have the most influence on model predictions</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="individual" className="space-y-4">
          <TabsList>
            <TabsTrigger value="individual">Individual Features</TabsTrigger>
            <TabsTrigger value="categories">Feature Categories</TabsTrigger>
          </TabsList>

          <TabsContent value="individual">
            {loading ? (
              <Skeleton className="h-[400px] w-full" />
            ) : (
              <ChartContainer
                config={{
                  importance: {
                    label: "Relative Importance",
                    color: "hsl(var(--chart-1))",
                  },
                }}
                className="h-[400px]"
              >
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={featureData.global}
                    layout="vertical"
                    margin={{ top: 5, right: 30, left: 120, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} />
                    <XAxis type="number" domain={[0, 100]} tickFormatter={(value) => `${value}%`} />
                    <YAxis type="category" dataKey="feature" width={110} tick={{ fontSize: 12 }} />
                    <ChartTooltip
                      content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                          return (
                            <div className="bg-background border rounded-md shadow-md p-2 text-sm">
                              <p className="font-medium">{payload[0].payload.feature}</p>
                              <p>Importance: {payload[0].payload.importance.toFixed(3)}</p>
                              <p>Relative: {payload[0].payload.normalized}%</p>
                            </div>
                          )
                        }
                        return null
                      }}
                    />
                    <Bar dataKey="normalized" fill="var(--color-importance)" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </ChartContainer>
            )}
          </TabsContent>

          <TabsContent value="categories">
            {loading ? (
              <Skeleton className="h-[400px] w-full" />
            ) : (
              <div className="h-[400px]">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={featureData.categories}
                      cx="50%"
                      cy="50%"
                      labelLine={true}
                      outerRadius={120}
                      fill="#8884d8"
                      dataKey="value"
                      label={({ name, value }) => `${name}: ${formatPercent(value)}`}
                    >
                      {featureData.categories.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value: number) => formatPercent(value)} />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
