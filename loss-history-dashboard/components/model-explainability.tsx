"use client"

import { useState, useEffect } from "react"
import { useWebSocketContext } from "@/components/websocket-provider"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { AlertCircle, BarChart3, LineChart, PieChart, CheckCircle, AlertTriangle, Activity, Clock } from "lucide-react"

// Define types for our explainability data
interface ShapValue {
  feature: string
  value: number
  impact: number
}

interface PredictionComparison {
  actual: number
  predicted: number
  difference: number
  timestamp: Date
  features: Record<string, number>
}

// Mock data for initial rendering
const initialShapValues: ShapValue[] = [
  { feature: "property_value", value: 0.25, impact: 0.15 },
  { feature: "claim_amount", value: 0.18, impact: 0.12 },
  { feature: "policy_age", value: 0.12, impact: 0.08 },
  { feature: "customer_age", value: 0.10, impact: 0.06 },
  { feature: "claim_frequency", value: 0.08, impact: 0.05 },
  { feature: "deductible", value: 0.07, impact: 0.04 },
  { feature: "coverage_type", value: 0.05, impact: 0.03 },
  { feature: "region", value: 0.04, impact: 0.02 },
  { feature: "vehicle_age", value: 0.03, impact: 0.02 },
  { feature: "credit_score", value: 0.02, impact: 0.01 }
]

const initialPredictionComparisons: PredictionComparison[] = [
  {
    actual: 1250.75,
    predicted: 1180.25,
    difference: 70.5,
    timestamp: new Date(Date.now() - 3600000),
    features: {
      property_value: 350000,
      claim_amount: 1200,
      policy_age: 3,
      customer_age: 42,
      claim_frequency: 1,
      deductible: 500,
      coverage_type: 2,
      region: 3,
      vehicle_age: 5,
      credit_score: 720
    }
  },
  {
    actual: 980.50,
    predicted: 1050.75,
    difference: -70.25,
    timestamp: new Date(Date.now() - 7200000),
    features: {
      property_value: 275000,
      claim_amount: 950,
      policy_age: 2,
      customer_age: 38,
      claim_frequency: 0,
      deductible: 750,
      coverage_type: 1,
      region: 2,
      vehicle_age: 3,
      credit_score: 680
    }
  },
  {
    actual: 1500.25,
    predicted: 1420.50,
    difference: 79.75,
    timestamp: new Date(Date.now() - 10800000),
    features: {
      property_value: 425000,
      claim_amount: 1450,
      policy_age: 5,
      customer_age: 45,
      claim_frequency: 2,
      deductible: 250,
      coverage_type: 3,
      region: 4,
      vehicle_age: 7,
      credit_score: 750
    }
  }
]

export function ModelExplainability() {
  const { lastMessage } = useWebSocketContext()
  const [shapValues, setShapValues] = useState<ShapValue[]>(initialShapValues)
  const [predictionComparisons, setPredictionComparisons] = useState<PredictionComparison[]>(initialPredictionComparisons)
  const [selectedModel, setSelectedModel] = useState("model1")
  const [activeTab, setActiveTab] = useState("shap")

  // Process incoming WebSocket messages
  useEffect(() => {
    if (!lastMessage) return

    // Handle SHAP values updates
    if (lastMessage.type === "shap_values" && lastMessage.data.modelId === selectedModel) {
      setShapValues(lastMessage.data.values)
    }
    
    // Handle prediction comparison updates
    if (lastMessage.type === "prediction_comparison" && lastMessage.data.modelId === selectedModel) {
      setPredictionComparisons(prev => [lastMessage.data, ...prev].slice(0, 10))
    }
  }, [lastMessage, selectedModel])

  // Calculate average prediction error
  const averageError = predictionComparisons.reduce((sum, pred) => sum + Math.abs(pred.difference), 0) / predictionComparisons.length

  // Calculate error distribution
  const errorDistribution = {
    underpredicted: predictionComparisons.filter(pred => pred.difference < 0).length,
    overpredicted: predictionComparisons.filter(pred => pred.difference > 0).length,
    accurate: predictionComparisons.filter(pred => Math.abs(pred.difference) < averageError * 0.1).length
  }

  // Helper function to get status badge color
  const getStatusColor = (status: string) => {
    switch (status) {
      case "healthy":
        return "bg-emerald-500/15 text-emerald-600"
      case "warning":
        return "bg-amber-500/15 text-amber-600"
      case "critical":
        return "bg-rose-500/15 text-rose-600"
      case "running":
        return "bg-sky-500/15 text-sky-600"
      case "completed":
        return "bg-emerald-500/15 text-emerald-600"
      case "failed":
        return "bg-rose-500/15 text-rose-600"
      case "idle":
        return "bg-gray-500/15 text-gray-600"
      default:
        return "bg-gray-500/15 text-gray-600"
    }
  }

  // Helper function to get status icon
  const getStatusIcon = (status: string) => {
    switch (status) {
      case "healthy":
      case "completed":
        return <CheckCircle className="h-4 w-4 text-emerald-600" />
      case "warning":
        return <AlertCircle className="h-4 w-4 text-amber-600" />
      case "critical":
      case "failed":
        return <AlertTriangle className="h-4 w-4 text-rose-600" />
      case "running":
        return <Activity className="h-4 w-4 text-sky-600" />
      case "idle":
        return <Clock className="h-4 w-4 text-gray-600" />
      default:
        return <Clock className="h-4 w-4 text-gray-600" />
    }
  }

  return (
    <div className="space-y-6 bg-gradient-to-b from-gray-50 to-gray-100 p-6 rounded-lg">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-800">Model Explainability</h2>
        <Select value={selectedModel} onValueChange={setSelectedModel}>
          <SelectTrigger className="w-[200px] bg-white border-gray-200">
            <SelectValue placeholder="Select model" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="gradient-boosting">Gradient Boosting</SelectItem>
            <SelectItem value="random-forest">Random Forest</SelectItem>
            <SelectItem value="xgboost">XGBoost</SelectItem>
            <SelectItem value="lightgbm">LightGBM</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid grid-cols-3 mb-4 bg-white border border-gray-200">
          <TabsTrigger value="shap" className="text-gray-700 data-[state=active]:text-gray-900">SHAP Values</TabsTrigger>
          <TabsTrigger value="predictions" className="text-gray-700 data-[state=active]:text-gray-900">Predictions</TabsTrigger>
          <TabsTrigger value="features" className="text-gray-700 data-[state=active]:text-gray-900">Feature Importance</TabsTrigger>
        </TabsList>

        <TabsContent value="shap" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card className="bg-white shadow-sm border border-gray-200">
              <CardHeader>
                <CardTitle className="text-gray-800">SHAP Values</CardTitle>
                <CardDescription className="text-gray-500">Feature impact on predictions</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {shapValues.map((value, index) => (
                    <div key={index} className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-gray-700">{value.feature}</span>
                        <span className={`font-medium ${
                          value.impact > 0 ? "text-emerald-600" : "text-rose-600"
                        }`}>
                          {value.impact > 0 ? "+" : ""}{value.impact.toFixed(4)}
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2.5">
                        <div 
                          className={`h-2.5 rounded-full ${
                            value.impact > 0 ? "bg-emerald-500" : "bg-rose-500"
                          }`} 
                          style={{ width: `${Math.abs(value.impact) * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card className="bg-white shadow-sm border border-gray-200">
              <CardHeader>
                <CardTitle className="text-gray-800">Feature Values</CardTitle>
                <CardDescription className="text-gray-500">Current feature values</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {shapValues.map((value, index) => (
                    <div key={index} className="flex justify-between items-center">
                      <span className="text-gray-700">{value.feature}</span>
                      <span className="text-gray-800">{value.value.toFixed(4)}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="predictions" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card className="bg-white shadow-sm border border-gray-200">
              <CardHeader>
                <CardTitle className="text-gray-800">Prediction Error</CardTitle>
                <CardDescription className="text-gray-500">Average error: {averageError.toFixed(4)}</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-700">Error Distribution</span>
                      <span className="text-gray-800">{errorDistribution.toFixed(2)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2.5">
                      <div 
                        className={`h-2.5 rounded-full ${
                          errorDistribution > 80 ? "bg-rose-500" : 
                          errorDistribution > 60 ? "bg-amber-500" : "bg-emerald-500"
                        }`} 
                        style={{ width: `${errorDistribution}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-white shadow-sm border border-gray-200">
              <CardHeader>
                <CardTitle className="text-gray-800">Recent Predictions</CardTitle>
                <CardDescription className="text-gray-500">Last 5 predictions</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {predictionComparisons.slice(0, 5).map((prediction, index) => (
                    <div key={index} className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-gray-700">Prediction {index + 1}</span>
                        <span className={`font-medium ${
                          Math.abs(prediction.difference) < 0.1 ? "text-emerald-600" : 
                          Math.abs(prediction.difference) < 0.3 ? "text-amber-600" : "text-rose-600"
                        }`}>
                          {prediction.difference > 0 ? "+" : ""}{prediction.difference.toFixed(4)}
                        </span>
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        <div>
                          <span className="text-gray-500">Actual: </span>
                          <span className="text-gray-800">{prediction.actual.toFixed(4)}</span>
                        </div>
                        <div>
                          <span className="text-gray-500">Predicted: </span>
                          <span className="text-gray-800">{prediction.predicted.toFixed(4)}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="features" className="space-y-4">
          <Card className="bg-white shadow-sm border border-gray-200">
            <CardHeader>
              <CardTitle className="text-gray-800">Feature Importance</CardTitle>
              <CardDescription className="text-gray-500">Relative importance of each feature</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {Object.entries(featureImportance)
                  .sort(([, a], [, b]) => b - a)
                  .map(([feature, importance], index) => (
                    <div key={index} className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-gray-700">{feature}</span>
                        <span className="text-gray-800">{(importance * 100).toFixed(2)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2.5">
                        <div 
                          className={`h-2.5 rounded-full ${
                            importance > 0.8 ? "bg-violet-500" : 
                            importance > 0.5 ? "bg-indigo-500" : "bg-sky-500"
                          }`} 
                          style={{ width: `${importance * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
} 