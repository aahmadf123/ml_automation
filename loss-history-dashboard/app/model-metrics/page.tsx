"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ExternalLink, RefreshCw, Download, AlertTriangle } from "lucide-react"
import { DashboardLayout } from "@/components/dashboard-layout"
import { ModelPerformance } from "@/components/model-performance"
import { ShapSummaryPlot } from "@/components/shap-summary-plot"
import { FeatureImportance } from "@/components/feature-importance"
import { ErrorBoundary } from "@/components/error-boundary"
import { LoadingSpinner } from "@/components/loading-spinner"
import { RealTimeMetrics } from "@/components/real-time-metrics"
import { useWebSocket } from "@/lib/websocket"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

const availableModels = [
  { id: "model1", name: "Loss Prediction Model" },
  { id: "model2", name: "Claim Amount Prediction" },
  { id: "model3", name: "Fraud Detection Model" },
  { id: "model4", name: "Risk Scoring Model" },
  { id: "model5", name: "Customer Churn Model" },
]

export default function ModelMetricsPage() {
  const [isLoading, setIsLoading] = useState(false)
  const [activeTab, setActiveTab] = useState("overview")
  const [selectedModel, setSelectedModel] = useState(availableModels[0].id)
  const [alerts, setAlerts] = useState<any[]>([])
  const { connected, lastMessage } = useWebSocket()

  useEffect(() => {
    if (lastMessage?.type === "alert" && lastMessage.data.modelId === selectedModel) {
      setAlerts(prev => [lastMessage.data, ...prev].slice(0, 5))
    }
  }, [lastMessage, selectedModel])

  const handleRefresh = async () => {
    setIsLoading(true)
    try {
      // Add your refresh logic here
      await new Promise(resolve => setTimeout(resolve, 1000))
    } finally {
      setIsLoading(false)
    }
  }

  const actions = (
    <>
      <Select value={selectedModel} onValueChange={setSelectedModel}>
        <SelectTrigger className="w-[200px]">
          <SelectValue placeholder="Select model" />
        </SelectTrigger>
        <SelectContent>
          {availableModels.map(model => (
            <SelectItem key={model.id} value={model.id}>
              {model.name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      <Button variant="outline" onClick={handleRefresh} disabled={isLoading}>
        <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? "animate-spin" : ""}`} />
        Refresh
      </Button>
      <Button variant="outline">
        <Download className="h-4 w-4 mr-2" />
        Export
      </Button>
      <Button variant="outline">
        <ExternalLink className="h-4 w-4 mr-2" />
        Open in MLflow
      </Button>
    </>
  )

  return (
    <DashboardLayout
      title="Model Metrics"
      description="Track and analyze the performance of your machine learning models"
      actions={actions}
    >
      <ErrorBoundary>
        {alerts.length > 0 && (
          <div className="space-y-2">
            {alerts.map((alert, index) => (
              <Alert key={index} variant={alert.severity === "high" ? "destructive" : "default"}>
                <AlertTriangle className="h-4 w-4" />
                <AlertTitle>{alert.title}</AlertTitle>
                <AlertDescription>{alert.message}</AlertDescription>
              </Alert>
            ))}
          </div>
        )}

        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
          <TabsList>
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="performance">Performance</TabsTrigger>
            <TabsTrigger value="explainability">Explainability</TabsTrigger>
            <TabsTrigger value="drift">Drift Analysis</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <RealTimeMetrics
                title="Model Performance"
                description="Real-time performance metrics"
                metricKey="performance"
                unit="%"
                color="#2563eb"
              />

              <RealTimeMetrics
                title="Memory Usage"
                description="Model memory consumption"
                metricKey="memory"
                unit="MB"
                color="#16a34a"
              />

              <RealTimeMetrics
                title="Inference Time"
                description="Average inference time"
                metricKey="inference_time"
                unit="ms"
                color="#dc2626"
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle>Feature Importance</CardTitle>
                  <CardDescription>Top contributing features</CardDescription>
                </CardHeader>
                <CardContent>
                  <ErrorBoundary>
                    <FeatureImportance modelId={selectedModel} />
                  </ErrorBoundary>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>SHAP Summary</CardTitle>
                  <CardDescription>Feature impact analysis</CardDescription>
                </CardHeader>
                <CardContent>
                  <ErrorBoundary>
                    <ShapSummaryPlot modelId={selectedModel} />
                  </ErrorBoundary>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="performance" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Detailed Performance Metrics</CardTitle>
                <CardDescription>Comprehensive model performance analysis</CardDescription>
              </CardHeader>
              <CardContent>
                <ErrorBoundary>
                  <ModelPerformance modelId={selectedModel} detailed />
                </ErrorBoundary>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="explainability" className="space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle>SHAP Analysis</CardTitle>
                  <CardDescription>Detailed feature impact analysis</CardDescription>
                </CardHeader>
                <CardContent>
                  <ErrorBoundary>
                    <ShapSummaryPlot modelId={selectedModel} detailed />
                  </ErrorBoundary>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Feature Importance</CardTitle>
                  <CardDescription>Detailed feature importance analysis</CardDescription>
                </CardHeader>
                <CardContent>
                  <ErrorBoundary>
                    <FeatureImportance modelId={selectedModel} detailed />
                  </ErrorBoundary>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="drift" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Drift Analysis</CardTitle>
                <CardDescription>Model drift detection and analysis</CardDescription>
              </CardHeader>
              <CardContent>
                <ErrorBoundary>
                  <RealTimeMetrics
                    title="Feature Drift"
                    description="Real-time feature distribution drift"
                    metricKey="feature_drift"
                    unit="%"
                    color="#9333ea"
                  />
                </ErrorBoundary>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </ErrorBoundary>
    </DashboardLayout>
  )
}
