"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Skeleton } from "@/components/ui/skeleton"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { ChartContainer } from "@/components/ui/chart"
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
  LineChart,
  Line,
} from "recharts"
import { Download, FileText, Search, RefreshCw, Layers, GitCompare } from "lucide-react"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"

// Mock model versions
const mockModelVersions = [
  { id: "model1-v5", name: "Loss Prediction v5", modelId: "model1", version: "5", date: "2023-04-15T10:23:45" },
  { id: "model1-v4", name: "Loss Prediction v4", modelId: "model1", version: "4", date: "2023-04-10T14:35:22" },
  { id: "model1-v3", name: "Loss Prediction v3", modelId: "model1", version: "3", date: "2023-04-05T09:12:18" },
  { id: "model1-v2", name: "Loss Prediction v2", modelId: "model1", version: "2", date: "2023-03-28T16:45:33" },
  { id: "model1-v1", name: "Loss Prediction v1", modelId: "model1", version: "1", date: "2023-03-20T11:30:45" },
  { id: "model2-v3", name: "Claim Amount v3", modelId: "model2", version: "3", date: "2023-04-14T16:42:12" },
  { id: "model2-v2", name: "Claim Amount v2", modelId: "model2", version: "2", date: "2023-04-08T12:15:33" },
  { id: "model2-v1", name: "Claim Amount v1", modelId: "model2", version: "1", date: "2023-04-01T09:45:21" },
  { id: "model3-v7", name: "Fraud Detection v7", modelId: "model3", version: "7", date: "2023-04-15T08:15:33" },
  { id: "model3-v6", name: "Fraud Detection v6", modelId: "model3", version: "6", date: "2023-04-12T14:22:45" },
]

// Mock predictions
const mockPredictions = [
  {
    id: "pred-001",
    claimId: "CLM-1234",
    probability: 0.82,
    timestamp: "2023-04-15T10:23:45",
    claimAmount: "$12,500",
    claimType: "Water Damage",
    policyHolder: "John Smith",
  },
  {
    id: "pred-002",
    claimId: "CLM-5678",
    probability: 0.45,
    timestamp: "2023-04-14T16:42:12",
    claimAmount: "$8,750",
    claimType: "Fire Damage",
    policyHolder: "Sarah Johnson",
  },
  {
    id: "pred-003",
    claimId: "CLM-9012",
    probability: 0.91,
    timestamp: "2023-04-15T08:15:33",
    claimAmount: "$22,300",
    claimType: "Storm Damage",
    policyHolder: "Michael Brown",
  },
]

// Mock SHAP comparison data
const generateMockShapComparisonData = (modelIds: string[]) => {
  const features = [
    "ClaimAmount",
    "PropertyAge",
    "ClaimHistory",
    "LocationRiskScore",
    "PropertyValue",
    "CoverageLevel",
    "ClaimType",
    "WeatherEvents",
    "SecuritySystem",
    "OwnershipDuration",
  ]

  return features.map((feature) => {
    const result: Record<string, any> = { feature }

    modelIds.forEach((modelId) => {
      // Generate a base value that's somewhat consistent across models
      const baseValue = Math.random() * 0.8 - 0.4 // Between -0.4 and 0.4

      // For newer versions, add some variation but keep the direction mostly the same
      const version = Number.parseInt(modelId.split("-v")[1])
      const versionFactor = 1 + version / 10 // Slight increase for newer versions

      result[modelId] = baseValue * versionFactor + (Math.random() * 0.2 - 0.1) // Add some noise
    })

    return result
  })
}

// Mock feature importance comparison data
const generateMockFeatureImportanceData = (modelIds: string[]) => {
  const features = [
    "ClaimAmount",
    "PropertyAge",
    "ClaimHistory",
    "LocationRiskScore",
    "PropertyValue",
    "CoverageLevel",
    "ClaimType",
    "WeatherEvents",
    "SecuritySystem",
    "OwnershipDuration",
  ]

  return features
    .map((feature) => {
      const result: Record<string, any> = { feature }

      modelIds.forEach((modelId) => {
        // Generate a base importance that's somewhat consistent
        const baseImportance = Math.random() * 0.3 + 0.05 // Between 0.05 and 0.35

        // For newer versions, add some variation
        const version = Number.parseInt(modelId.split("-v")[1])
        const versionFactor = 1 + (version - 1) * 0.05 // Slight adjustment for versions

        result[modelId] = baseImportance * versionFactor
      })

      // Sort by average importance
      return result
    })
    .sort((a, b) => {
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

// Mock prediction comparison data
const generateMockPredictionComparisonData = (modelIds: string[], predictionId: string) => {
  // Base probability that will be adjusted for each model version
  const baseProbability = Math.random() * 0.5 + 0.3 // Between 0.3 and 0.8

  return modelIds.map((modelId) => {
    const version = Number.parseInt(modelId.split("-v")[1])
    // Newer versions tend to have slightly different predictions
    // But maintain some consistency
    const versionAdjustment = (version / 20) * (Math.random() > 0.5 ? 1 : -1)
    const probability = Math.max(0.01, Math.min(0.99, baseProbability + versionAdjustment))

    return {
      modelId,
      predictionId,
      probability,
      timestamp: new Date(Date.now() - version * 86400000).toISOString(), // Older versions have older timestamps
    }
  })
}

// Mock explanation difference metrics
const generateMockExplanationDifferenceMetrics = (modelIds: string[]) => {
  if (modelIds.length < 2) return []

  const results = []

  // Generate pairwise comparisons
  for (let i = 0; i < modelIds.length; i++) {
    for (let j = i + 1; j < modelIds.length; j++) {
      const model1 = modelIds[i]
      const model2 = modelIds[j]

      // Generate some metrics
      const shapDifference = Math.random() * 0.3 + 0.05
      const featureRankCorrelation = Math.random() * 0.4 + 0.6
      const topFeatureOverlap = Math.floor(Math.random() * 3) + 7 // Between 7-10 out of 10
      const predictionDifference = Math.random() * 0.15

      results.push({
        model1,
        model2,
        shapDifference,
        featureRankCorrelation,
        topFeatureOverlap,
        predictionDifference,
      })
    }
  }

  return results
}

// Generate time series data for explanation stability
const generateExplanationStabilityData = (modelIds: string[]) => {
  const timePoints = 10 // 10 time points
  const result = []

  for (let i = 0; i < timePoints; i++) {
    const entry: Record<string, any> = {
      timePoint: i,
      date: new Date(Date.now() - (timePoints - i) * 86400000 * 3).toISOString().split("T")[0],
    }

    modelIds.forEach((modelId) => {
      const version = Number.parseInt(modelId.split("-v")[1])
      // Base stability score
      let stability = 0.7 + version / 20

      // Add some noise and trends
      stability += Math.sin(i / 2) * 0.05 // Cyclical component
      stability += (i / timePoints) * 0.1 // Trend component (improving over time)
      stability += Math.random() * 0.1 - 0.05 // Random noise

      // Ensure it's between 0 and 1
      stability = Math.max(0, Math.min(1, stability))

      entry[modelId] = stability
    })

    result.push(entry)
  }

  return result
}

interface ModelExplanationComparisonProps {
  initialModelIds?: string[]
  initialPredictionId?: string
}

export function ModelExplanationComparison({
  initialModelIds = ["model1-v5", "model1-v4", "model1-v3"],
  initialPredictionId = "pred-001",
}: ModelExplanationComparisonProps) {
  const [loading, setLoading] = useState(true)
  const [modelVersions, setModelVersions] = useState(mockModelVersions)
  const [selectedModelIds, setSelectedModelIds] = useState<string[]>(initialModelIds)
  const [predictions, setPredictions] = useState(mockPredictions)
  const [selectedPredictionId, setSelectedPredictionId] = useState(initialPredictionId)
  const [shapComparisonData, setShapComparisonData] = useState<any[]>([])
  const [featureImportanceData, setFeatureImportanceData] = useState<any[]>([])
  const [predictionComparisonData, setPredictionComparisonData] = useState<any[]>([])
  const [explanationDifferenceMetrics, setExplanationDifferenceMetrics] = useState<any[]>([])
  const [explanationStabilityData, setExplanationStabilityData] = useState<any[]>([])
  const [searchQuery, setSearchQuery] = useState("")
  const [modelFilter, setModelFilter] = useState<string | null>(null)

  useEffect(() => {
    // Simulate API call to get model versions and predictions
    const fetchInitialData = async () => {
      // In a real implementation, this would fetch from your API
      // const modelVersionsResponse = await fetch('/api/model-versions');
      // const predictionsResponse = await fetch('/api/predictions');

      // Simulate loading delay
      setTimeout(() => {
        setModelVersions(mockModelVersions)
        setPredictions(mockPredictions)
        setLoading(false)
      }, 1000)
    }

    fetchInitialData()
  }, [])

  useEffect(() => {
    if (selectedModelIds.length > 0 && selectedPredictionId) {
      setLoading(true)

      // Simulate API calls to get comparison data
      const fetchComparisonData = async () => {
        // In a real implementation, these would be separate API calls
        // const shapResponse = await fetch(`/api/explanations/shap-comparison?modelIds=${selectedModelIds.join(',')}&predictionId=${selectedPredictionId}`);
        // const importanceResponse = await fetch(`/api/explanations/feature-importance-comparison?modelIds=${selectedModelIds.join(',')}`);
        // etc.

        // Generate mock data
        const mockShapData = generateMockShapComparisonData(selectedModelIds)
        const mockImportanceData = generateMockFeatureImportanceData(selectedModelIds)
        const mockPredictionData = generateMockPredictionComparisonData(selectedModelIds, selectedPredictionId)
        const mockDifferenceMetrics = generateMockExplanationDifferenceMetrics(selectedModelIds)
        const mockStabilityData = generateExplanationStabilityData(selectedModelIds)

        // Simulate loading delay
        setTimeout(() => {
          setShapComparisonData(mockShapData)
          setFeatureImportanceData(mockImportanceData)
          setPredictionComparisonData(mockPredictionData)
          setExplanationDifferenceMetrics(mockDifferenceMetrics)
          setExplanationStabilityData(mockStabilityData)
          setLoading(false)
        }, 1200)
      }

      fetchComparisonData()
    }
  }, [selectedModelIds, selectedPredictionId])

  const toggleModelSelection = (modelId: string) => {
    if (selectedModelIds.includes(modelId)) {
      // Don't allow deselecting if only one model is selected
      if (selectedModelIds.length > 1) {
        setSelectedModelIds(selectedModelIds.filter((id) => id !== modelId))
      }
    } else {
      setSelectedModelIds([...selectedModelIds, modelId])
    }
  }

  const filteredModelVersions = modelVersions.filter((model) => {
    // Apply search filter
    const matchesSearch =
      searchQuery === "" ||
      model.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      model.version.includes(searchQuery)

    // Apply model filter
    const matchesModelFilter = modelFilter === null || model.modelId === modelFilter

    return matchesSearch && matchesModelFilter
  })

  const getModelName = (modelId: string) => {
    const model = modelVersions.find((m) => m.id === modelId)
    return model ? model.name : modelId
  }

  const getModelColor = (index: number) => {
    const colors = [
      "hsl(var(--chart-1))",
      "hsl(var(--chart-2))",
      "hsl(var(--chart-3))",
      "hsl(var(--chart-4))",
      "hsl(var(--chart-5))",
    ]
    return colors[index % colors.length]
  }

  return (
    <Card className="shadow-md">
      <CardHeader>
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <CardTitle>Model Explanation Comparison</CardTitle>
            <CardDescription>Compare explanations across different model versions</CardDescription>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={() => setLoading(true)}>
              <RefreshCw className="h-4 w-4 mr-1" />
              Refresh
            </Button>
            <Button size="sm">
              <Download className="h-4 w-4 mr-1" />
              Export
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Model Selection Panel */}
          <Card className="md:col-span-1">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg">Select Model Versions</CardTitle>
              <CardDescription>Choose 2-5 model versions to compare</CardDescription>

              <div className="flex gap-2 mt-2">
                <div className="relative flex-1">
                  <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                  <Input
                    type="search"
                    placeholder="Search models..."
                    className="pl-8"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                  />
                </div>
                <Select value={modelFilter || ""} onValueChange={(value) => setModelFilter(value || null)}>
                  <SelectTrigger className="w-[130px]">
                    <SelectValue placeholder="Filter models" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Models</SelectItem>
                    <SelectItem value="model1">Loss Prediction</SelectItem>
                    <SelectItem value="model2">Claim Amount</SelectItem>
                    <SelectItem value="model3">Fraud Detection</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardHeader>
            <CardContent className="pb-2">
              <ScrollArea className="h-[300px] pr-4">
                <div className="space-y-4">
                  {loading
                    ? Array(5)
                        .fill(0)
                        .map((_, i) => (
                          <div key={i} className="flex items-center space-x-2">
                            <Skeleton className="h-4 w-4 rounded" />
                            <Skeleton className="h-4 flex-1" />
                          </div>
                        ))
                    : filteredModelVersions.map((model) => (
                        <div key={model.id} className="flex items-center space-x-2">
                          <Checkbox
                            id={`model-${model.id}`}
                            checked={selectedModelIds.includes(model.id)}
                            onCheckedChange={() => toggleModelSelection(model.id)}
                            disabled={selectedModelIds.length >= 5 && !selectedModelIds.includes(model.id)}
                          />
                          <Label
                            htmlFor={`model-${model.id}`}
                            className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 flex-1"
                          >
                            {model.name}
                          </Label>
                          <span className="text-xs text-muted-foreground">
                            {new Date(model.date).toLocaleDateString()}
                          </span>
                        </div>
                      ))}
                </div>
              </ScrollArea>

              <div className="mt-4">
                <p className="text-sm text-muted-foreground mb-2">
                  Selected: {selectedModelIds.length} of {modelVersions.length}
                </p>
                <div className="flex flex-wrap gap-2">
                  {selectedModelIds.map((modelId, index) => (
                    <Badge key={modelId} variant="outline" className="bg-primary/10">
                      {getModelName(modelId)}
                    </Badge>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Prediction Selection Panel */}
          <Card className="md:col-span-2">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg">Select Prediction to Explain</CardTitle>
              <CardDescription>Choose a prediction to compare explanations across model versions</CardDescription>
            </CardHeader>
            <CardContent>
              {loading ? (
                <Skeleton className="h-[300px] w-full" />
              ) : (
                <div className="space-y-4">
                  <Select value={selectedPredictionId} onValueChange={setSelectedPredictionId}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select a prediction" />
                    </SelectTrigger>
                    <SelectContent>
                      {predictions.map((pred) => (
                        <SelectItem key={pred.id} value={pred.id}>
                          {pred.claimId} - {pred.policyHolder} ({pred.claimType})
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>

                  {selectedPredictionId && (
                    <div className="border rounded-md p-4 bg-muted/30">
                      <h3 className="font-medium mb-2">Selected Prediction Details</h3>
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                        {predictions
                          .filter((p) => p.id === selectedPredictionId)
                          .map((pred) => (
                            <div key={pred.id}>
                              <div>
                                <p className="text-sm text-muted-foreground">Claim ID</p>
                                <p className="font-medium">{pred.claimId}</p>
                              </div>
                              <div className="mt-2">
                                <p className="text-sm text-muted-foreground">Policy Holder</p>
                                <p className="font-medium">{pred.policyHolder}</p>
                              </div>
                              <div className="mt-2">
                                <p className="text-sm text-muted-foreground">Claim Type</p>
                                <p className="font-medium">{pred.claimType}</p>
                              </div>
                              <div className="mt-2">
                                <p className="text-sm text-muted-foreground">Claim Amount</p>
                                <p className="font-medium">{pred.claimAmount}</p>
                              </div>
                              <div className="mt-2">
                                <p className="text-sm text-muted-foreground">Date</p>
                                <p className="font-medium">{new Date(pred.timestamp).toLocaleDateString()}</p>
                              </div>
                              <div className="mt-2">
                                <p className="text-sm text-muted-foreground">Latest Prediction</p>
                                <p className="font-medium">
                                  <span className={pred.probability > 0.5 ? "text-red-500" : "text-green-500"}>
                                    {(pred.probability * 100).toFixed(1)}%
                                  </span>{" "}
                                  probability of loss
                                </p>
                              </div>
                            </div>
                          ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Comparison Tabs */}
        <Tabs defaultValue="shap" className="space-y-4">
          <TabsList className="grid grid-cols-2 md:grid-cols-5 w-full">
            <TabsTrigger value="shap">SHAP Values</TabsTrigger>
            <TabsTrigger value="importance">Feature Importance</TabsTrigger>
            <TabsTrigger value="predictions">Prediction Changes</TabsTrigger>
            <TabsTrigger value="metrics">Difference Metrics</TabsTrigger>
            <TabsTrigger value="stability">Explanation Stability</TabsTrigger>
          </TabsList>

          {/* SHAP Values Comparison */}
          <TabsContent value="shap" className="space-y-4">
            <div className="text-sm text-muted-foreground">
              Compare how SHAP values for each feature have changed across model versions. This shows how the impact of
              each feature on predictions has evolved.
            </div>
            {loading ? (
              <Skeleton className="h-[500px] w-full" />
            ) : (
              <ChartContainer className="h-[500px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={shapComparisonData}
                    layout="vertical"
                    margin={{ top: 20, right: 30, left: 120, bottom: 5 }}
                  >
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
                                    <span>{getModelName(entry.dataKey)}:</span>
                                    <span
                                      className={`font-medium ${
                                        Number(entry.value) > 0 ? "text-green-500" : "text-red-500"
                                      }`}
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
                    <Legend formatter={(value) => getModelName(value)} />
                    <ReferenceLine x={0} stroke="#666" />
                    {selectedModelIds.map((modelId, index) => (
                      <Bar key={modelId} dataKey={modelId} fill={getModelColor(index)} radius={[0, 4, 4, 0]} />
                    ))}
                  </BarChart>
                </ResponsiveContainer>
              </ChartContainer>
            )}
            <div className="p-4 border rounded-md bg-muted/30">
              <h3 className="font-medium mb-2">Key Insights:</h3>
              <ul className="list-disc pl-5 space-y-1 text-sm">
                <li>
                  <span className="font-medium">Feature Shift:</span> The impact of ClaimAmount has increased in newer
                  versions, while PropertyAge has decreased in importance.
                </li>
                <li>
                  <span className="font-medium">Sign Changes:</span> WeatherEvents changed from negative to positive
                  impact in the latest version, indicating a fundamental shift in how this feature is interpreted.
                </li>
                <li>
                  <span className="font-medium">Consistency:</span> ClaimHistory and LocationRiskScore have maintained
                  consistent impact across versions, suggesting stable feature relationships.
                </li>
              </ul>
            </div>
          </TabsContent>

          {/* Feature Importance Comparison */}
          <TabsContent value="importance" className="space-y-4">
            <div className="text-sm text-muted-foreground">
              Compare how feature importance rankings have changed across model versions. This shows which features have
              become more or less important to the model over time.
            </div>
            {loading ? (
              <Skeleton className="h-[500px] w-full" />
            ) : (
              <ChartContainer className="h-[500px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={featureImportanceData}
                    layout="vertical"
                    margin={{ top: 20, right: 30, left: 120, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} />
                    <XAxis type="number" domain={[0, 0.4]} tickFormatter={(value) => value.toFixed(2)} />
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
                                    <span>{getModelName(entry.dataKey)}:</span>
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
                    <Legend formatter={(value) => getModelName(value)} />
                    {selectedModelIds.map((modelId, index) => (
                      <Bar key={modelId} dataKey={modelId} fill={getModelColor(index)} radius={[0, 4, 4, 0]} />
                    ))}
                  </BarChart>
                </ResponsiveContainer>
              </ChartContainer>
            )}
            <div className="p-4 border rounded-md bg-muted/30">
              <h3 className="font-medium mb-2">Key Insights:</h3>
              <ul className="list-disc pl-5 space-y-1 text-sm">
                <li>
                  <span className="font-medium">Ranking Changes:</span> ClaimAmount has consistently been the most
                  important feature, but its relative importance has increased in newer versions.
                </li>
                <li>
                  <span className="font-medium">New Features:</span> LocationRiskScore has gained importance in newer
                  versions, suggesting model improvements in geographic risk assessment.
                </li>
                <li>
                  <span className="font-medium">Diminished Features:</span> Some features like SecuritySystem have
                  decreased in importance, indicating they may be less predictive in newer models.
                </li>
              </ul>
            </div>
          </TabsContent>

          {/* Prediction Changes */}
          <TabsContent value="predictions" className="space-y-4">
            <div className="text-sm text-muted-foreground">
              Compare how the prediction probability has changed across model versions for the selected claim.
            </div>
            {loading ? (
              <Skeleton className="h-[400px] w-full" />
            ) : (
              <ChartContainer className="h-[400px]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={predictionComparisonData.sort(
                      (a, b) => Number.parseInt(a.modelId.split("-v")[1]) - Number.parseInt(b.modelId.split("-v")[1]),
                    )}
                    margin={{ top: 20, right: 30, left: 20, bottom: 70 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="modelId"
                      tick={{ fontSize: 12 }}
                      tickFormatter={(value) => value.split("-v")[1]}
                      label={{ value: "Model Version", position: "bottom", offset: 40 }}
                    />
                    <YAxis
                      domain={[0, 1]}
                      tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                      label={{ value: "Prediction Probability", angle: -90, position: "insideLeft" }}
                    />
                    <Tooltip
                      formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, "Probability"]}
                      labelFormatter={(value) => `${getModelName(value)}`}
                    />
                    <ReferenceLine y={0.5} stroke="#666" strokeDasharray="3 3" />
                    <Line
                      type="monotone"
                      dataKey="probability"
                      stroke="hsl(var(--primary))"
                      strokeWidth={2}
                      dot={{ r: 6 }}
                      activeDot={{ r: 8 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </ChartContainer>
            )}
            <div className="p-4 border rounded-md bg-muted/30">
              <h3 className="font-medium mb-2">Prediction Evolution:</h3>
              <p className="text-sm mb-2">
                The prediction probability for this claim has{" "}
                {predictionComparisonData.length > 1 &&
                predictionComparisonData[predictionComparisonData.length - 1].probability >
                  predictionComparisonData[0].probability
                  ? "increased"
                  : "decreased"}{" "}
                across model versions.
              </p>
              <ul className="list-disc pl-5 space-y-1 text-sm">
                <li>
                  <span className="font-medium">Threshold Crossing:</span> The prediction{" "}
                  {predictionComparisonData.some((d) => d.probability < 0.5) &&
                  predictionComparisonData.some((d) => d.probability >= 0.5)
                    ? "crosses the 0.5 threshold between versions, potentially changing the decision"
                    : "remains consistently on the same side of the threshold across all versions"}
                  .
                </li>
                <li>
                  <span className="font-medium">Confidence Change:</span> Newer models{" "}
                  {predictionComparisonData.length > 1 &&
                  Math.abs(predictionComparisonData[predictionComparisonData.length - 1].probability - 0.5) >
                    Math.abs(predictionComparisonData[0].probability - 0.5)
                    ? "show higher confidence (further from 0.5)"
                    : "show lower confidence (closer to 0.5)"}{" "}
                  in their predictions.
                </li>
              </ul>
            </div>
          </TabsContent>

          {/* Difference Metrics */}
          <TabsContent value="metrics" className="space-y-4">
            <div className="text-sm text-muted-foreground">
              Quantitative metrics showing how explanations differ between model versions.
            </div>
            {loading ? (
              <Skeleton className="h-[400px] w-full" />
            ) : (
              <div className="border rounded-md">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Model Comparison</TableHead>
                      <TableHead>SHAP Difference</TableHead>
                      <TableHead>Feature Rank Correlation</TableHead>
                      <TableHead>Top Feature Overlap</TableHead>
                      <TableHead>Prediction Difference</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {explanationDifferenceMetrics.map((metric, index) => (
                      <TableRow key={index}>
                        <TableCell className="font-medium">
                          {getModelName(metric.model1)} vs {getModelName(metric.model2)}
                        </TableCell>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <span>{metric.shapDifference.toFixed(3)}</span>
                            <div className="w-16 bg-muted rounded-full h-2">
                              <div
                                className={`h-2 rounded-full ${
                                  metric.shapDifference < 0.1
                                    ? "bg-green-500"
                                    : metric.shapDifference < 0.2
                                      ? "bg-yellow-500"
                                      : "bg-red-500"
                                }`}
                                style={{ width: `${Math.min(metric.shapDifference * 100, 100)}%` }}
                              />
                            </div>
                          </div>
                        </TableCell>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <span>{metric.featureRankCorrelation.toFixed(2)}</span>
                            <div className="w-16 bg-muted rounded-full h-2">
                              <div
                                className={`h-2 rounded-full ${
                                  metric.featureRankCorrelation > 0.8
                                    ? "bg-green-500"
                                    : metric.featureRankCorrelation > 0.6
                                      ? "bg-yellow-500"
                                      : "bg-red-500"
                                }`}
                                style={{ width: `${metric.featureRankCorrelation * 100}%` }}
                              />
                            </div>
                          </div>
                        </TableCell>
                        <TableCell>{metric.topFeatureOverlap}/10</TableCell>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <span>{(metric.predictionDifference * 100).toFixed(1)}%</span>
                            <div className="w-16 bg-muted rounded-full h-2">
                              <div
                                className={`h-2 rounded-full ${
                                  metric.predictionDifference < 0.05
                                    ? "bg-green-500"
                                    : metric.predictionDifference < 0.1
                                      ? "bg-yellow-500"
                                      : "bg-red-500"
                                }`}
                                style={{ width: `${Math.min(metric.predictionDifference * 200, 100)}%` }}
                              />
                            </div>
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            )}
            <div className="p-4 border rounded-md bg-muted/30">
              <h3 className="font-medium mb-2">Metrics Explained:</h3>
              <ul className="list-disc pl-5 space-y-1 text-sm">
                <li>
                  <span className="font-medium">SHAP Difference:</span> Average absolute difference in SHAP values
                  across features. Lower values indicate more similar explanations.
                </li>
                <li>
                  <span className="font-medium">Feature Rank Correlation:</span> Spearman correlation between feature
                  importance rankings. Higher values indicate more consistent feature rankings.
                </li>
                <li>
                  <span className="font-medium">Top Feature Overlap:</span> Number of features that appear in the top 10
                  most important features for both models.
                </li>
                <li>
                  <span className="font-medium">Prediction Difference:</span> Absolute difference in prediction
                  probability between models.
                </li>
              </ul>
            </div>
          </TabsContent>

          {/* Explanation Stability */}
          <TabsContent value="stability" className="space-y-4">
            <div className="text-sm text-muted-foreground">
              Track how explanation stability has evolved over time for different model versions.
            </div>
            {loading ? (
              <Skeleton className="h-[400px] w-full" />
            ) : (
              <ChartContainer className="h-[400px]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={explanationStabilityData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis
                      domain={[0, 1]}
                      tickFormatter={(value) => value.toFixed(1)}
                      label={{ value: "Explanation Stability", angle: -90, position: "insideLeft" }}
                    />
                    <Tooltip formatter={(value: number) => [value.toFixed(3), "Stability Score"]} />
                    <Legend formatter={(value) => getModelName(value)} />
                    {selectedModelIds.map((modelId, index) => (
                      <Line
                        key={modelId}
                        type="monotone"
                        dataKey={modelId}
                        stroke={getModelColor(index)}
                        strokeWidth={2}
                        dot={{ r: 4 }}
                        activeDot={{ r: 6 }}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </ChartContainer>
            )}
            <div className="p-4 border rounded-md bg-muted/30">
              <h3 className="font-medium mb-2">Stability Analysis:</h3>
              <p className="text-sm mb-2">
                Explanation stability measures how consistent a model's explanations are over time and across similar
                inputs.
              </p>
              <ul className="list-disc pl-5 space-y-1 text-sm">
                <li>
                  <span className="font-medium">Trend Analysis:</span> Newer model versions generally show improved
                  stability over time, indicating more robust and consistent explanations.
                </li>
                <li>
                  <span className="font-medium">Volatility:</span> Some models show periodic drops in stability,
                  potentially corresponding to data drift or concept shift events.
                </li>
                <li>
                  <span className="font-medium">Version Comparison:</span> Version 5 shows the highest overall
                  stability, suggesting that the latest model improvements have led to more reliable explanations.
                </li>
              </ul>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
      <CardFooter className="flex justify-between">
        <Button variant="outline">
          <GitCompare className="mr-2 h-4 w-4" />
          Compare More Models
        </Button>
        <div className="flex gap-2">
          <Button variant="outline">
            <FileText className="mr-2 h-4 w-4" />
            Generate Report
          </Button>
          <Button>
            <Layers className="mr-2 h-4 w-4" />
            Save Comparison
          </Button>
        </div>
      </CardFooter>
    </Card>
  )
}
