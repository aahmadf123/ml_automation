"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Download, Info } from "lucide-react"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts"

// Mock prediction data
const mockPredictions = [
  {
    id: "pred-001",
    claimId: "CLM-1234",
    input: {
      claimAmount: 12500,
      propertyAge: 15,
      claimHistory: 3,
      locationRiskScore: 8.2,
      propertyValue: 450000,
      coverageLevel: "Premium",
      claimType: "Water Damage",
      weatherEvents: "Recent flooding",
      securitySystem: "Basic",
      ownershipDuration: 7,
    },
    predictions: {
      model1: { probability: 0.82, confidence: 0.91 },
      model2: { probability: 0.78, confidence: 0.88 },
      model3: { probability: 0.85, confidence: 0.92 },
    },
  },
  {
    id: "pred-002",
    claimId: "CLM-5678",
    input: {
      claimAmount: 8200,
      propertyAge: 22,
      claimHistory: 1,
      locationRiskScore: 5.4,
      propertyValue: 320000,
      coverageLevel: "Standard",
      claimType: "Fire Damage",
      weatherEvents: "None",
      securitySystem: "Advanced",
      ownershipDuration: 12,
    },
    predictions: {
      model1: { probability: 0.45, confidence: 0.82 },
      model2: { probability: 0.52, confidence: 0.79 },
      model3: { probability: 0.38, confidence: 0.85 },
    },
  },
  {
    id: "pred-003",
    claimId: "CLM-9012",
    input: {
      claimAmount: 18700,
      propertyAge: 8,
      claimHistory: 4,
      locationRiskScore: 9.1,
      propertyValue: 580000,
      coverageLevel: "Premium",
      claimType: "Storm Damage",
      weatherEvents: "Hurricane",
      securitySystem: "Standard",
      ownershipDuration: 5,
    },
    predictions: {
      model1: { probability: 0.91, confidence: 0.94 },
      model2: { probability: 0.88, confidence: 0.92 },
      model3: { probability: 0.93, confidence: 0.95 },
    },
  },
  {
    id: "pred-004",
    claimId: "CLM-3456",
    input: {
      claimAmount: 5300,
      propertyAge: 30,
      claimHistory: 0,
      locationRiskScore: 3.2,
      propertyValue: 280000,
      coverageLevel: "Basic",
      claimType: "Theft",
      weatherEvents: "None",
      securitySystem: "None",
      ownershipDuration: 18,
    },
    predictions: {
      model1: { probability: 0.23, confidence: 0.75 },
      model2: { probability: 0.28, confidence: 0.72 },
      model3: { probability: 0.19, confidence: 0.78 },
    },
  },
  {
    id: "pred-005",
    claimId: "CLM-7890",
    input: {
      claimAmount: 9800,
      propertyAge: 12,
      claimHistory: 2,
      locationRiskScore: 6.8,
      propertyValue: 410000,
      coverageLevel: "Standard",
      claimType: "Water Damage",
      weatherEvents: "Heavy Rain",
      securitySystem: "Standard",
      ownershipDuration: 9,
    },
    predictions: {
      model1: { probability: 0.67, confidence: 0.85 },
      model2: { probability: 0.72, confidence: 0.83 },
      model3: { probability: 0.65, confidence: 0.87 },
    },
  },
]

interface PredictionComparisonProps {
  modelIds: string[]
  modelNames?: Record<string, string>
}

export function PredictionComparison({
  modelIds,
  modelNames = {
    model1: "Loss Prediction",
    model2: "Claim Amount",
    model3: "Fraud Detection",
  },
}: PredictionComparisonProps) {
  const [loading, setLoading] = useState(true)
  const [predictions, setPredictions] = useState<any[]>([])
  const [selectedPrediction, setSelectedPrediction] = useState("")

  useEffect(() => {
    // Simulate API call to get predictions
    const fetchPredictions = async () => {
      // In a real implementation, this would fetch from your API
      // const response = await fetch(`/api/predictions?modelIds=${modelIds.join(',')}`);
      // const data = await response.json();

      // Simulate loading delay
      setTimeout(() => {
        // Filter predictions to only include requested models
        const filteredPredictions = mockPredictions.map((pred) => ({
          ...pred,
          predictions: Object.fromEntries(Object.entries(pred.predictions).filter(([key]) => modelIds.includes(key))),
        }))

        setPredictions(filteredPredictions)
        if (filteredPredictions.length > 0) {
          setSelectedPrediction(filteredPredictions[0].id)
        }
        setLoading(false)
      }, 1000)
    }

    if (modelIds.length > 0) {
      fetchPredictions()
    }
  }, [modelIds])

  // Get the selected prediction data
  const selectedPredictionData = predictions.find((p) => p.id === selectedPrediction)

  // Prepare chart data for the selected prediction
  const preparePredictionChartData = () => {
    if (!selectedPredictionData) return []

    return Object.entries(selectedPredictionData.predictions).map(([modelId, data]) => ({
      modelId,
      modelName: modelNames[modelId] || modelId,
      probability: (data as any).probability,
      confidence: (data as any).confidence,
    }))
  }

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
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <CardTitle>Prediction Comparison</CardTitle>
            <CardDescription>Compare how different models predict the same input data</CardDescription>
          </div>

          <div className="w-full md:w-64">
            {loading ? (
              <Skeleton className="h-10 w-full" />
            ) : (
              <Select value={selectedPrediction} onValueChange={setSelectedPrediction}>
                <SelectTrigger>
                  <SelectValue placeholder="Select a prediction" />
                </SelectTrigger>
                <SelectContent>
                  {predictions.map((pred) => (
                    <SelectItem key={pred.id} value={pred.id}>
                      {pred.claimId}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="space-y-4">
            <Skeleton className="h-20 w-full" />
            <Skeleton className="h-[400px] w-full" />
          </div>
        ) : selectedPredictionData ? (
          <div className="space-y-6">
            <Card className="bg-muted/50">
              <CardContent className="p-4">
                <h3 className="text-lg font-semibold mb-2">Input Features</h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
                  {Object.entries(selectedPredictionData.input).map(([key, value]) => (
                    <div key={key}>
                      <h4 className="text-sm font-medium text-muted-foreground capitalize">
                        {key.replace(/([A-Z])/g, " $1").trim()}
                      </h4>
                      <p className="text-sm font-semibold">{value}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div>
                <h3 className="text-lg font-semibold mb-4">Prediction Probabilities</h3>
                <div className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={preparePredictionChartData()}
                      layout="vertical"
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} />
                      <XAxis type="number" domain={[0, 1]} tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
                      <YAxis type="category" dataKey="modelName" width={120} tick={{ fontSize: 12 }} />
                      <Tooltip formatter={(value) => `${(Number(value) * 100).toFixed(1)}%`} />
                      <Bar dataKey="probability" name="Probability" radius={[0, 4, 4, 0]}>
                        {preparePredictionChartData().map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={getModelColor(entry.modelId)} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-4">Confidence Levels</h3>
                <div className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={preparePredictionChartData()}
                      layout="vertical"
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} />
                      <XAxis type="number" domain={[0, 1]} tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
                      <YAxis type="category" dataKey="modelName" width={120} tick={{ fontSize: 12 }} />
                      <Tooltip formatter={(value) => `${(Number(value) * 100).toFixed(1)}%`} />
                      <Bar dataKey="confidence" name="Confidence" radius={[0, 4, 4, 0]}>
                        {preparePredictionChartData().map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={getModelColor(entry.modelId)} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            <div>
              <h3 className="text-lg font-semibold mb-4">Detailed Comparison</h3>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Model</TableHead>
                    <TableHead>Probability</TableHead>
                    <TableHead>Confidence</TableHead>
                    <TableHead>Prediction</TableHead>
                    <TableHead>Difference from Average</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {Object.entries(selectedPredictionData.predictions).map(([modelId, data]) => {
                    const avgProbability =
                      Object.values(selectedPredictionData.predictions).reduce(
                        (sum, val) => sum + (val as any).probability,
                        0,
                      ) / Object.values(selectedPredictionData.predictions).length

                    const diff = (data as any).probability - avgProbability

                    return (
                      <TableRow key={modelId}>
                        <TableCell className="font-medium">{modelNames[modelId] || modelId}</TableCell>
                        <TableCell>{((data as any).probability * 100).toFixed(1)}%</TableCell>
                        <TableCell>{((data as any).confidence * 100).toFixed(1)}%</TableCell>
                        <TableCell>
                          <span className={(data as any).probability > 0.5 ? "text-red-500" : "text-green-500"}>
                            {(data as any).probability > 0.5 ? "High Risk" : "Low Risk"}
                          </span>
                        </TableCell>
                        <TableCell>
                          <span className={diff > 0 ? "text-red-500" : "text-green-500"}>
                            {diff > 0 ? "+" : ""}
                            {(diff * 100).toFixed(1)}%
                          </span>
                        </TableCell>
                      </TableRow>
                    )
                  })}
                </TableBody>
              </Table>
            </div>

            <div className="flex justify-end space-x-2">
              <Button variant="outline">
                <Info className="mr-2 h-4 w-4" />
                View Details
              </Button>
              <Button>
                <Download className="mr-2 h-4 w-4" />
                Export Comparison
              </Button>
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center h-[400px]">
            <p className="text-muted-foreground">No prediction data available</p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
