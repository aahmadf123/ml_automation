"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Skeleton } from "@/components/ui/skeleton"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { ShapSummaryPlot } from "./shap-summary-plot"
import { Input } from "@/components/ui/input"
import { Search, Download, FileText, BarChart3, Info } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { ShapWaterfallPlot } from "./shap-waterfall-plot"
import { ShapForceValuePlot } from "./shap-force-value-plot"

// Mock prediction data
const mockPredictions = [
  {
    id: "pred-001",
    claimId: "CLM-1234",
    probability: 0.82,
    timestamp: "2023-04-15T10:23:45",
    claimAmount: "$12,500",
    claimType: "Water Damage",
    policyHolder: "John Smith",
    propertyAddress: "123 Main St, Anytown, USA",
  },
  {
    id: "pred-002",
    claimId: "CLM-5678",
    probability: 0.45,
    timestamp: "2023-04-14T16:42:12",
    claimAmount: "$8,750",
    claimType: "Fire Damage",
    policyHolder: "Sarah Johnson",
    propertyAddress: "456 Oak Ave, Somewhere, USA",
  },
  {
    id: "pred-003",
    claimId: "CLM-9012",
    probability: 0.91,
    timestamp: "2023-04-15T08:15:33",
    claimAmount: "$22,300",
    claimType: "Storm Damage",
    policyHolder: "Michael Brown",
    propertyAddress: "789 Pine Rd, Elsewhere, USA",
  },
  {
    id: "pred-004",
    claimId: "CLM-3456",
    probability: 0.23,
    timestamp: "2023-04-13T22:10:05",
    claimAmount: "$5,100",
    claimType: "Theft",
    policyHolder: "Emily Davis",
    propertyAddress: "101 Elm St, Nowhere, USA",
  },
  {
    id: "pred-005",
    claimId: "CLM-7890",
    probability: 0.67,
    timestamp: "2023-04-14T14:55:21",
    claimAmount: "$15,800",
    claimType: "Water Damage",
    policyHolder: "Robert Wilson",
    propertyAddress: "202 Maple Dr, Anyplace, USA",
  },
]

// Mock feature values for a specific prediction
const mockFeatureValues = [
  {
    feature: "ClaimAmount",
    value: "$12,500",
    impact: 0.85,
    direction: "positive",
    description: "Higher claim amounts are associated with higher loss probability",
  },
  {
    feature: "PropertyAge",
    value: "15 years",
    impact: -0.72,
    direction: "negative",
    description: "Older properties tend to have different risk profiles",
  },
  {
    feature: "ClaimHistory",
    value: "3 prior claims",
    impact: 0.68,
    direction: "positive",
    description: "More prior claims indicate higher risk",
  },
  {
    feature: "LocationRiskScore",
    value: "8.2/10",
    impact: 0.61,
    direction: "positive",
    description: "Higher risk locations correlate with higher loss probability",
  },
  {
    feature: "PropertyValue",
    value: "$450,000",
    impact: -0.58,
    direction: "negative",
    description: "Higher value properties show different loss patterns",
  },
  {
    feature: "CoverageLevel",
    value: "Premium",
    impact: 0.52,
    direction: "positive",
    description: "Premium coverage correlates with certain claim types",
  },
  {
    feature: "ClaimType",
    value: "Water Damage",
    impact: -0.48,
    direction: "negative",
    description: "Water damage claims have specific loss patterns",
  },
  {
    feature: "WeatherEvents",
    value: "Recent flooding",
    impact: 0.45,
    direction: "positive",
    description: "Recent weather events increase certain risks",
  },
  {
    feature: "SecuritySystem",
    value: "Basic",
    impact: -0.42,
    direction: "negative",
    description: "Security systems affect certain loss types",
  },
  {
    feature: "OwnershipDuration",
    value: "7 years",
    impact: 0.38,
    direction: "positive",
    description: "Ownership duration correlates with claim patterns",
  },
]

// Mock similar cases for comparison
const mockSimilarCases = [
  { claimId: "CLM-2468", similarity: 0.92, probability: 0.79, outcome: "Loss", amount: "$11,800" },
  { claimId: "CLM-1357", similarity: 0.87, probability: 0.84, outcome: "Loss", amount: "$13,200" },
  { claimId: "CLM-8642", similarity: 0.81, probability: 0.76, outcome: "Loss", amount: "$10,500" },
  { claimId: "CLM-9753", similarity: 0.78, probability: 0.45, outcome: "No Loss", amount: "$12,700" },
  { claimId: "CLM-3141", similarity: 0.75, probability: 0.88, outcome: "Loss", amount: "$14,300" },
]

interface PredictionExplainerProps {
  modelId?: string
  initialPredictionId?: string
}

export function PredictionExplainer({ modelId = "model1", initialPredictionId }: PredictionExplainerProps) {
  const [loading, setLoading] = useState(true)
  const [predictions, setPredictions] = useState(mockPredictions)
  const [selectedPrediction, setSelectedPrediction] = useState(initialPredictionId || "")
  const [featureValues, setFeatureValues] = useState(mockFeatureValues)
  const [searchQuery, setSearchQuery] = useState("")
  const [similarCases, setSimilarCases] = useState(mockSimilarCases)
  const [activeFeature, setActiveFeature] = useState<string | null>(null)

  useEffect(() => {
    // Simulate API call to get predictions
    const fetchPredictions = async () => {
      // In a real implementation, this would fetch from your API
      // const response = await fetch(`/api/predictions?modelId=${modelId}`);
      // const data = await response.json();

      // Simulate loading delay
      setTimeout(() => {
        setPredictions(mockPredictions)
        if (!selectedPrediction && mockPredictions.length > 0) {
          setSelectedPrediction(mockPredictions[0].id)
        }
        setLoading(false)
      }, 1000)
    }

    fetchPredictions()
  }, [modelId, selectedPrediction])

  useEffect(() => {
    // Fetch feature values for the selected prediction
    if (selectedPrediction) {
      // In a real implementation, this would fetch from your API
      // const response = await fetch(`/api/prediction-features?predictionId=${selectedPrediction}`);
      // const data = await response.json();

      // Using mock data for now
      setFeatureValues(mockFeatureValues)
      setSimilarCases(mockSimilarCases)
    }
  }, [selectedPrediction])

  const getImpactColor = (impact: number) => {
    const absImpact = Math.abs(impact)
    if (impact > 0) {
      if (absImpact > 0.7) return "text-green-600"
      if (absImpact > 0.4) return "text-green-500"
      return "text-green-400"
    } else {
      if (absImpact > 0.7) return "text-red-600"
      if (absImpact > 0.4) return "text-red-500"
      return "text-red-400"
    }
  }

  const getImpactDescription = (impact: number) => {
    const absImpact = Math.abs(impact)
    const direction = impact > 0 ? "positive" : "negative"

    if (absImpact > 0.7) return `High ${direction} impact`
    if (absImpact > 0.4) return `Moderate ${direction} impact`
    return `Low ${direction} impact`
  }

  const filteredPredictions = predictions.filter(
    (pred) =>
      pred.claimId.toLowerCase().includes(searchQuery.toLowerCase()) ||
      pred.policyHolder.toLowerCase().includes(searchQuery.toLowerCase()) ||
      pred.claimType.toLowerCase().includes(searchQuery.toLowerCase()),
  )

  const selectedPredictionData = predictions.find((p) => p.id === selectedPrediction)

  return (
    <Card className="shadow-md">
      <CardHeader className="pb-2">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <CardTitle>Prediction Explainer</CardTitle>
            <CardDescription>Understand why the model made specific predictions</CardDescription>
          </div>

          <div className="w-full md:w-96 flex gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                type="search"
                placeholder="Search claims..."
                className="pl-8"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
            {loading ? (
              <Skeleton className="h-10 w-40" />
            ) : (
              <Select value={selectedPrediction} onValueChange={setSelectedPrediction}>
                <SelectTrigger className="w-40">
                  <SelectValue placeholder="Select claim" />
                </SelectTrigger>
                <SelectContent>
                  {filteredPredictions.map((pred) => (
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
        ) : (
          <div className="space-y-6">
            {selectedPredictionData && (
              <Card className="bg-muted/50">
                <CardContent className="p-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    <div>
                      <h4 className="text-sm font-medium text-muted-foreground">Claim ID</h4>
                      <p className="text-lg font-semibold">{selectedPredictionData.claimId}</p>
                      <p className="text-sm text-muted-foreground">{selectedPredictionData.policyHolder}</p>
                    </div>
                    <div>
                      <h4 className="text-sm font-medium text-muted-foreground">Prediction</h4>
                      <p className="text-lg font-semibold">
                        <span className={selectedPredictionData.probability > 0.5 ? "text-red-500" : "text-green-500"}>
                          {(selectedPredictionData.probability * 100).toFixed(1)}%
                        </span>{" "}
                        probability of loss
                      </p>
                      <Badge
                        variant={
                          selectedPredictionData.probability > 0.7
                            ? "destructive"
                            : selectedPredictionData.probability > 0.4
                              ? "warning"
                              : "success"
                        }
                      >
                        {selectedPredictionData.probability > 0.7
                          ? "High Risk"
                          : selectedPredictionData.probability > 0.4
                            ? "Medium Risk"
                            : "Low Risk"}
                      </Badge>
                    </div>
                    <div>
                      <h4 className="text-sm font-medium text-muted-foreground">Claim Details</h4>
                      <p className="text-lg font-semibold">{selectedPredictionData.claimAmount}</p>
                      <p className="text-sm text-muted-foreground">{selectedPredictionData.claimType}</p>
                    </div>
                    <div>
                      <h4 className="text-sm font-medium text-muted-foreground">Timestamp</h4>
                      <p className="text-lg font-semibold">
                        {new Date(selectedPredictionData.timestamp).toLocaleDateString()}
                      </p>
                      <p className="text-sm text-muted-foreground">
                        {new Date(selectedPredictionData.timestamp).toLocaleTimeString()}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            <Tabs defaultValue="shap" className="space-y-4">
              <TabsList className="grid grid-cols-4">
                <TabsTrigger value="shap">SHAP Summary</TabsTrigger>
                <TabsTrigger value="waterfall">SHAP Waterfall</TabsTrigger>
                <TabsTrigger value="features">Feature Values</TabsTrigger>
                <TabsTrigger value="similar">Similar Cases</TabsTrigger>
              </TabsList>

              <TabsContent value="shap">
                <ShapSummaryPlot modelId={modelId} predictionId={selectedPrediction} />
              </TabsContent>

              <TabsContent value="waterfall">
                <ShapWaterfallPlot modelId={modelId} predictionId={selectedPrediction} />
              </TabsContent>

              <TabsContent value="features">
                <div className="border rounded-md">
                  <div className="grid grid-cols-12 bg-muted p-3 rounded-t-md">
                    <div className="col-span-3 font-medium">Feature</div>
                    <div className="col-span-3 font-medium">Value</div>
                    <div className="col-span-3 font-medium">Impact</div>
                    <div className="col-span-3 font-medium">Description</div>
                  </div>
                  <ScrollArea className="h-[400px]">
                    <div className="divide-y">
                      {featureValues.map((feature, index) => (
                        <div
                          key={index}
                          className={`grid grid-cols-12 p-3 hover:bg-muted/50 cursor-pointer ${activeFeature === feature.feature ? "bg-muted/50" : ""}`}
                          onClick={() => setActiveFeature(activeFeature === feature.feature ? null : feature.feature)}
                        >
                          <div className="col-span-3 font-medium">{feature.feature}</div>
                          <div className="col-span-3">{feature.value}</div>
                          <div className={`col-span-3 ${getImpactColor(feature.impact)}`}>
                            {getImpactDescription(feature.impact)}
                          </div>
                          <div className="col-span-3 flex items-center">
                            <span className="truncate mr-2">{feature.description}</span>
                            <TooltipProvider>
                              <Tooltip>
                                <TooltipTrigger>
                                  <Info className="h-4 w-4 text-muted-foreground" />
                                </TooltipTrigger>
                                <TooltipContent>
                                  <p className="max-w-xs">{feature.description}</p>
                                </TooltipContent>
                              </Tooltip>
                            </TooltipProvider>
                          </div>
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </div>
                {activeFeature && (
                  <div className="mt-4">
                    <ShapForceValuePlot feature={activeFeature} modelId={modelId} predictionId={selectedPrediction} />
                  </div>
                )}
              </TabsContent>

              <TabsContent value="similar">
                <div className="border rounded-md">
                  <div className="grid grid-cols-12 bg-muted p-3 rounded-t-md">
                    <div className="col-span-3 font-medium">Claim ID</div>
                    <div className="col-span-3 font-medium">Similarity</div>
                    <div className="col-span-2 font-medium">Probability</div>
                    <div className="col-span-2 font-medium">Outcome</div>
                    <div className="col-span-2 font-medium">Amount</div>
                  </div>
                  <div className="divide-y">
                    {similarCases.map((claim, index) => (
                      <div key={index} className="grid grid-cols-12 p-3 hover:bg-muted/50">
                        <div className="col-span-3">{claim.claimId}</div>
                        <div className="col-span-3">
                          <div className="flex items-center">
                            <div className="w-full bg-muted rounded-full h-2 mr-2">
                              <div
                                className="bg-primary h-2 rounded-full"
                                style={{ width: `${claim.similarity * 100}%` }}
                              />
                            </div>
                            <span>{(claim.similarity * 100).toFixed(0)}%</span>
                          </div>
                        </div>
                        <div className="col-span-2">
                          <span className={claim.probability > 0.5 ? "text-red-500" : "text-green-500"}>
                            {(claim.probability * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div className="col-span-2">
                          <Badge variant={claim.outcome === "Loss" ? "destructive" : "success"}>{claim.outcome}</Badge>
                        </div>
                        <div className="col-span-2">{claim.amount}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </TabsContent>
            </Tabs>

            <div className="flex justify-end space-x-2">
              <Button variant="outline">
                <FileText className="mr-2 h-4 w-4" />
                Generate Report
              </Button>
              <Button variant="outline">
                <BarChart3 className="mr-2 h-4 w-4" />
                Compare
              </Button>
              <Button>
                <Download className="mr-2 h-4 w-4" />
                Export Explanation
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
