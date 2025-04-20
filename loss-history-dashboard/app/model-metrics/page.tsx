"use client"

import { DashboardHeader } from "@/components/dashboard-header"
import { PageHeader } from "@/components/page-header"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ModelPerformance } from "@/components/model-performance"
import { ShapSummaryPlot } from "@/components/shap-summary-plot"
import { FeatureImportance } from "@/components/feature-importance"
import { Button } from "@/components/ui/button"
import { ExternalLink } from "lucide-react"

export default function ModelMetricsPage() {
  return (
    <div className="flex flex-col min-h-screen">
      <DashboardHeader />
      <div className="flex-1 p-6 space-y-6 md:ml-64">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <PageHeader
            title="Model Metrics"
            description="Track and analyze the performance of your machine learning models"
          />

          <Button variant="outline" className="w-full md:w-auto">
            <ExternalLink className="h-4 w-4 mr-2" />
            Open in MLflow
          </Button>
        </div>

        <Tabs defaultValue="overview" className="space-y-4">
          <TabsList>
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="accuracy">Accuracy</TabsTrigger>
            <TabsTrigger value="explainability">Explainability</TabsTrigger>
            <TabsTrigger value="comparison">Model Comparison</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-4">
            <ModelPerformance />
          </TabsContent>

          <TabsContent value="accuracy" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Accuracy Metrics</CardTitle>
                <CardDescription>Detailed accuracy metrics for your models</CardDescription>
              </CardHeader>
              <CardContent className="h-[400px] flex items-center justify-center">
                <p className="text-muted-foreground">Accuracy metrics visualization will appear here</p>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="explainability" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <ShapSummaryPlot />
              <FeatureImportance />
            </div>

            <div className="flex justify-end">
              <Button asChild>
                <a href="/model-explainability">View Full Explainability Dashboard</a>
              </Button>
            </div>
          </TabsContent>

          <TabsContent value="comparison" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Model Comparison</CardTitle>
                <CardDescription>Compare performance across different model versions</CardDescription>
              </CardHeader>
              <CardContent className="h-[400px] flex items-center justify-center">
                <p className="text-muted-foreground">Model comparison visualization will appear here</p>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
