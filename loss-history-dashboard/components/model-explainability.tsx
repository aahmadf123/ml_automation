"use client"

import { useState, useEffect } from "react"
import { useWebSocketContext } from "@/components/websocket-provider"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { AlertCircle, BarChart3, LineChart, PieChart } from "lucide-react"

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

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Model Explainability</h2>
        <div className="flex items-center space-x-2">
          <span className="text-sm text-gray-500">Model:</span>
          <Select value={selectedModel} onValueChange={setSelectedModel}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Select model" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="model1">Gradient Boosting</SelectItem>
              <SelectItem value="model2">Random Forest</SelectItem>
              <SelectItem value="model3">XGBoost</SelectItem>
              <SelectItem value="model4">Neural Network</SelectItem>
              <SelectItem value="model5">LightGBM</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid grid-cols-3 mb-4">
          <TabsTrigger value="shap">SHAP Values</TabsTrigger>
          <TabsTrigger value="predictions">Actual vs. Predicted</TabsTrigger>
          <TabsTrigger value="feature-importance">Feature Importance</TabsTrigger>
        </TabsList>

        <TabsContent value="shap" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>SHAP Values</CardTitle>
              <CardDescription>Feature impact on model predictions</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {shapValues.map((item, index) => (
                  <div key={item.feature} className="space-y-1">
                    <div className="flex justify-between">
                      <span className="font-medium">{item.feature}</span>
                      <span className="text-sm text-gray-500">Impact: {item.impact.toFixed(4)}</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2.5">
                      <div 
                        className={`h-2.5 rounded-full ${
                          item.impact > 0.1 ? "bg-blue-500" : 
                          item.impact > 0.05 ? "bg-green-500" : "bg-gray-500"
                        }`} 
                        style={{ width: `${item.impact * 100}%` }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>SHAP Summary</CardTitle>
              <CardDescription>Overall feature importance based on SHAP values</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-64 flex items-center justify-center">
                <div className="text-center">
                  <BarChart3 className="h-12 w-12 mx-auto text-gray-400" />
                  <p className="mt-2 text-sm text-gray-500">SHAP summary visualization would appear here</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="predictions" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Average Error</CardTitle>
                <CardDescription>Mean absolute difference</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">${averageError.toFixed(2)}</div>
                <div className="text-sm text-gray-500 mt-1">
                  {averageError < 50 ? "Excellent accuracy" : averageError < 100 ? "Good accuracy" : "Needs improvement"}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Error Distribution</CardTitle>
                <CardDescription>Prediction accuracy breakdown</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>Underpredicted</span>
                    <Badge variant="outline" className="bg-red-500/10 text-red-500">
                      {errorDistribution.underpredicted}
                    </Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Overpredicted</span>
                    <Badge variant="outline" className="bg-blue-500/10 text-blue-500">
                      {errorDistribution.overpredicted}
                    </Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Accurate</span>
                    <Badge variant="outline" className="bg-green-500/10 text-green-500">
                      {errorDistribution.accurate}
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Prediction Trend</CardTitle>
                <CardDescription>Recent prediction accuracy</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-32 flex items-center justify-center">
                  <div className="text-center">
                    <LineChart className="h-12 w-12 mx-auto text-gray-400" />
                    <p className="mt-2 text-sm text-gray-500">Prediction trend visualization would appear here</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Recent Predictions</CardTitle>
              <CardDescription>Actual vs. predicted values with feature contributions</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-2">Timestamp</th>
                      <th className="text-right py-2">Actual</th>
                      <th className="text-right py-2">Predicted</th>
                      <th className="text-right py-2">Difference</th>
                      <th className="text-right py-2">Accuracy</th>
                    </tr>
                  </thead>
                  <tbody>
                    {predictionComparisons.map((pred, index) => (
                      <tr key={index} className="border-b">
                        <td className="py-2">{pred.timestamp.toLocaleTimeString()}</td>
                        <td className="text-right">${pred.actual.toFixed(2)}</td>
                        <td className="text-right">${pred.predicted.toFixed(2)}</td>
                        <td className={`text-right ${pred.difference > 0 ? "text-red-500" : "text-green-500"}`}>
                          {pred.difference > 0 ? "+" : ""}{pred.difference.toFixed(2)}
                        </td>
                        <td className="text-right">
                          <Badge 
                            variant="outline" 
                            className={
                              Math.abs(pred.difference) < averageError * 0.1 ? "bg-green-500/10 text-green-500" :
                              Math.abs(pred.difference) < averageError * 0.5 ? "bg-yellow-500/10 text-yellow-500" :
                              "bg-red-500/10 text-red-500"
                            }
                          >
                            {Math.abs(pred.difference) < averageError * 0.1 ? "Excellent" :
                             Math.abs(pred.difference) < averageError * 0.5 ? "Good" : "Poor"}
                          </Badge>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="feature-importance" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Feature Importance</CardTitle>
              <CardDescription>Relative importance of features in model predictions</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-64 flex items-center justify-center">
                <div className="text-center">
                  <PieChart className="h-12 w-12 mx-auto text-gray-400" />
                  <p className="mt-2 text-sm text-gray-500">Feature importance visualization would appear here</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Feature Correlations</CardTitle>
              <CardDescription>Correlation between features and predictions</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-2">Feature</th>
                      <th className="text-right py-2">Correlation</th>
                      <th className="text-right py-2">Impact</th>
                    </tr>
                  </thead>
                  <tbody>
                    {shapValues.map((item, index) => (
                      <tr key={item.feature} className="border-b">
                        <td className="py-2">{item.feature}</td>
                        <td className="text-right">
                          <Badge 
                            variant="outline" 
                            className={
                              item.impact > 0.1 ? "bg-blue-500/10 text-blue-500" :
                              item.impact > 0.05 ? "bg-green-500/10 text-green-500" :
                              "bg-gray-500/10 text-gray-500"
                            }
                          >
                            {item.impact > 0.1 ? "Strong" :
                             item.impact > 0.05 ? "Moderate" : "Weak"}
                          </Badge>
                        </td>
                        <td className="text-right">{item.impact.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
} 