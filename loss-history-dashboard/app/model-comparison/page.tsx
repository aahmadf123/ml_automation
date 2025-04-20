"use client"

import { Badge } from "@/components/ui/badge"

import { useState } from "react"
import { DashboardHeader } from "@/components/dashboard-header"
import { PageHeader } from "@/components/page-header"
import { Card, CardContent } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Download, RefreshCw } from "lucide-react"
import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"
import { FeatureImportanceComparison } from "@/components/model-comparison/feature-importance-comparison"
import { ShapComparison } from "@/components/model-comparison/shap-comparison"
import { PerformanceMetricsComparison } from "@/components/model-comparison/performance-metrics-comparison"
import { PredictionComparison } from "@/components/model-comparison/prediction-comparison"
import { FeatureRankingComparison } from "@/components/model-comparison/feature-ranking-comparison"

const availableModels = [
  { id: "model1", name: "Loss Prediction Model" },
  { id: "model2", name: "Claim Amount Prediction" },
  { id: "model3", name: "Fraud Detection Model" },
  { id: "model4", name: "Risk Scoring Model" },
  { id: "model5", name: "Customer Churn Model" },
]

export default function ModelComparisonPage() {
  const [selectedModels, setSelectedModels] = useState<string[]>(["model1", "model2", "model3"])

  const modelNames = Object.fromEntries(availableModels.map((model) => [model.id, model.name]))

  const toggleModel = (modelId: string) => {
    if (selectedModels.includes(modelId)) {
      // Don't allow deselecting if only one model is selected
      if (selectedModels.length > 1) {
        setSelectedModels(selectedModels.filter((id) => id !== modelId))
      }
    } else {
      setSelectedModels([...selectedModels, modelId])
    }
  }

  return (
    <div className="flex flex-col min-h-screen">
      <DashboardHeader />
      <div className="flex-1 p-6 space-y-6 md:ml-64">
        <PageHeader
          title="Model Comparison"
          description="Compare explainability and performance across different models"
        />

        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div className="flex gap-2">
            <Button variant="outline">
              <RefreshCw className="mr-2 h-4 w-4" />
              Refresh Data
            </Button>
            <Button>
              <Download className="mr-2 h-4 w-4" />
              Export Report
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-6">
          <Card className="md:col-span-1">
            <CardContent className="p-4">
              <h3 className="text-lg font-semibold mb-4">Select Models</h3>

              <div className="space-y-3">
                {availableModels.map((model) => (
                  <div key={model.id} className="flex items-center space-x-2">
                    <Checkbox
                      id={`model-${model.id}`}
                      checked={selectedModels.includes(model.id)}
                      onCheckedChange={() => toggleModel(model.id)}
                    />
                    <Label
                      htmlFor={`model-${model.id}`}
                      className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                    >
                      {model.name}
                    </Label>
                  </div>
                ))}
              </div>

              <div className="mt-6">
                <p className="text-sm text-muted-foreground mb-2">
                  Selected: {selectedModels.length} of {availableModels.length}
                </p>
                <div className="flex flex-wrap gap-2">
                  {selectedModels.map((modelId) => (
                    <Badge key={modelId} variant="outline" className="bg-primary/10">
                      {modelNames[modelId]}
                    </Badge>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>

          <div className="md:col-span-4 space-y-6">
            <Tabs defaultValue="feature-importance" className="space-y-4">
              <TabsList className="grid grid-cols-2 md:grid-cols-5 w-full">
                <TabsTrigger value="feature-importance">Feature Importance</TabsTrigger>
                <TabsTrigger value="shap">SHAP Values</TabsTrigger>
                <TabsTrigger value="performance">Performance</TabsTrigger>
                <TabsTrigger value="predictions">Predictions</TabsTrigger>
                <TabsTrigger value="rankings">Feature Rankings</TabsTrigger>
              </TabsList>

              <TabsContent value="feature-importance" className="space-y-4">
                <FeatureImportanceComparison modelIds={selectedModels} modelNames={modelNames} />
              </TabsContent>

              <TabsContent value="shap" className="space-y-4">
                <ShapComparison modelIds={selectedModels} modelNames={modelNames} />
              </TabsContent>

              <TabsContent value="performance" className="space-y-4">
                <PerformanceMetricsComparison modelIds={selectedModels} modelNames={modelNames} />
              </TabsContent>

              <TabsContent value="predictions" className="space-y-4">
                <PredictionComparison modelIds={selectedModels} modelNames={modelNames} />
              </TabsContent>

              <TabsContent value="rankings" className="space-y-4">
                <FeatureRankingComparison modelIds={selectedModels} modelNames={modelNames} />
              </TabsContent>
            </Tabs>
          </div>
        </div>
      </div>
    </div>
  )
}
