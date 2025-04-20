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
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

// Mock performance metrics data
const mockPerformanceMetrics = {
  model1: {
    accuracy: 0.85,
    precision: 0.82,
    recall: 0.79,
    f1Score: 0.8,
    auc: 0.88,
    logLoss: 0.32,
    mse: 0.15,
    mae: 0.12,
  },
  model2: {
    accuracy: 0.88,
    precision: 0.85,
    recall: 0.82,
    f1Score: 0.83,
    auc: 0.91,
    logLoss: 0.28,
    mse: 0.12,
    mae: 0.1,
  },
  model3: {
    accuracy: 0.82,
    precision: 0.79,
    recall: 0.85,
    f1Score: 0.82,
    auc: 0.86,
    logLoss: 0.35,
    mse: 0.18,
    mae: 0.14,
  },
  model4: {
    accuracy: 0.86,
    precision: 0.83,
    recall: 0.8,
    f1Score: 0.81,
    auc: 0.89,
    logLoss: 0.3,
    mse: 0.14,
    mae: 0.11,
  },
  model5: {
    accuracy: 0.84,
    precision: 0.81,
    recall: 0.83,
    f1Score: 0.82,
    auc: 0.87,
    logLoss: 0.33,
    mse: 0.16,
    mae: 0.13,
  },
}

// Prepare data for bar chart comparison
const prepareBarChartData = (modelData: Record<string, any>, metrics: string[]) => {
  return metrics.map((metric) => {
    const result: Record<string, any> = { metric }

    Object.entries(modelData).forEach(([modelId, modelMetrics]) => {
      result[modelId] = modelMetrics[metric as keyof typeof modelMetrics]
    })

    return result
  })
}

// Prepare data for radar chart
const prepareRadarData = (modelData: Record<string, any>, metrics: string[]) => {
  const result: any[] = []

  Object.entries(modelData).forEach(([modelId, modelMetrics]) => {
    const radarData = metrics.map((metric) => ({
      metric,
      value: modelMetrics[metric as keyof typeof modelMetrics],
      fullMark: 1.0,
    }))

    result.push({
      modelId,
      data: radarData,
    })
  })

  return result
}

interface PerformanceMetricsComparisonProps {
  modelIds: string[]
  modelNames?: Record<string, string>
}

export function PerformanceMetricsComparison({
  modelIds,
  modelNames = {
    model1: "Loss Prediction",
    model2: "Claim Amount",
    model3: "Fraud Detection",
    model4: "Risk Scoring",
    model5: "Customer Churn",
  },
}: PerformanceMetricsComparisonProps) {
  const [loading, setLoading] = useState(true)
  const [modelData, setModelData] = useState<Record<string, any>>({})
  const [barChartData, setBarChartData] = useState<any[]>([])
  const [radarData, setRadarData] = useState<any[]>([])

  // Define which metrics to show
  const classificationMetrics = ["accuracy", "precision", "recall", "f1Score", "auc"]
  const regressionMetrics = ["mse", "mae", "logLoss"]

  useEffect(() => {
    // Simulate API call to get performance metrics for multiple models
    const fetchPerformanceMetrics = async () => {
      // In a real implementation, this would fetch from your API
      // const promises = modelIds.map(id => fetch(`/api/performance-metrics?modelId=${id}`).then(res => res.json()))
      // const results = await Promise.all(promises)

      // Simulate loading delay
      setTimeout(() => {
        // Filter mock data to only include requested models
        const filteredData = Object.fromEntries(
          Object.entries(mockPerformanceMetrics).filter(([key]) => modelIds.includes(key)),
        )

        setModelData(filteredData)
        setBarChartData(prepareBarChartData(filteredData, classificationMetrics))
        setRadarData(prepareRadarData(filteredData, classificationMetrics))
        setLoading(false)
      }, 1000)
    }

    if (modelIds.length > 0) {
      fetchPerformanceMetrics()
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

  // Format metric names for display
  const formatMetricName = (metric: string) => {
    switch (metric) {
      case "accuracy":
        return "Accuracy"
      case "precision":
        return "Precision"
      case "recall":
        return "Recall"
      case "f1Score":
        return "F1 Score"
      case "auc":
        return "AUC"
      case "logLoss":
        return "Log Loss"
      case "mse":
        return "MSE"
      case "mae":
        return "MAE"
      default:
        return metric
    }
  }

  return (
    <Card className="shadow-md">
      <CardHeader className="pb-2">
        <CardTitle>Performance Metrics Comparison</CardTitle>
        <CardDescription>Compare accuracy and other metrics across different models</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="bar" className="space-y-4">
          <TabsList>
            <TabsTrigger value="bar">Bar Chart</TabsTrigger>
            <TabsTrigger value="radar">Radar Chart</TabsTrigger>
          </TabsList>

          <TabsContent value="bar">
            {loading ? (
              <Skeleton className="h-[400px] w-full" />
            ) : (
              <div className="h-[400px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={barChartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="metric" tickFormatter={formatMetricName} />
                    <YAxis domain={[0, 1]} tickFormatter={(value) => value.toFixed(1)} />
                    <Tooltip
                      content={({ active, payload, label }) => {
                        if (active && payload && payload.length) {
                          return (
                            <div className="bg-background border rounded-md shadow-md p-3 text-sm">
                              <p className="font-medium mb-1">{formatMetricName(label)}</p>
                              <div className="space-y-1">
                                {payload.map((entry, index) => (
                                  <div key={index} className="flex items-center gap-2">
                                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: entry.color }} />
                                    <span>{modelNames[entry.dataKey] || entry.dataKey}:</span>
                                    <span className="font-medium">{Number(entry.value).toFixed(3)}</span>
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
                      <Bar key={modelId} dataKey={modelId} fill={getModelColor(modelId)} radius={[4, 4, 0, 0]} />
                    ))}
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </TabsContent>

          <TabsContent value="radar">
            {loading ? (
              <Skeleton className="h-[400px] w-full" />
            ) : (
              <div className="h-[400px]">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData[0]?.data}>
                    <PolarGrid />
                    <PolarAngleAxis dataKey="metric" tickFormatter={formatMetricName} />
                    <PolarRadiusAxis angle={90} domain={[0, 1]} />
                    {radarData.map((entry, index) => (
                      <Radar
                        key={entry.modelId}
                        name={modelNames[entry.modelId] || entry.modelId}
                        dataKey="value"
                        stroke={getModelColor(entry.modelId)}
                        fill={getModelColor(entry.modelId)}
                        fillOpacity={0.2}
                      />
                    ))}
                    <Legend />
                    <Tooltip
                      content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                          const dataPoint = payload[0].payload
                          return (
                            <div className="bg-background border rounded-md shadow-md p-3 text-sm">
                              <p className="font-medium mb-1">{formatMetricName(dataPoint.payload.metric)}</p>
                              <div className="space-y-1">
                                {payload.map((entry, index) => (
                                  <div key={index} className="flex items-center gap-2">
                                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: entry.color }} />
                                    <span>{entry.name}:</span>
                                    <span className="font-medium">{Number(entry.value).toFixed(3)}</span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )
                        }
                        return null
                      }}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
