"use client"

import { DashboardHeader } from "@/components/dashboard-header"
import { PageHeader } from "@/components/page-header"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ShapSummaryPlot } from "@/components/shap-summary-plot"
import { FeatureImportance } from "@/components/feature-importance"
import { PredictionExplainer } from "@/components/prediction-explainer"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { GitCompare, Search, FileText, Download } from "lucide-react"
import { useState } from "react"
import { Input } from "@/components/ui/input"

export default function ModelExplainabilityPage() {
  const [selectedModel, setSelectedModel] = useState("model1")
  const [searchQuery, setSearchQuery] = useState("")

  return (
    <div className="flex flex-col min-h-screen">
      <DashboardHeader />
      <div className="flex-1 p-6 space-y-6 md:ml-64">
        <PageHeader
          title="Model Explainability"
          description="Understand how your models make predictions and which features matter most"
        />

        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div className="flex flex-col md:flex-row gap-4">
            <div className="w-full md:w-64">
              <Select value={selectedModel} onValueChange={setSelectedModel}>
                <SelectTrigger>
                  <SelectValue placeholder="Select a model" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="model1">Model 1 (Loss Prediction)</SelectItem>
                  <SelectItem value="model2">Model 2 (Claim Amount)</SelectItem>
                  <SelectItem value="model3">Model 3 (Fraud Detection)</SelectItem>
                  <SelectItem value="model4">Model 4 (Risk Scoring)</SelectItem>
                  <SelectItem value="model5">Model 5 (Customer Churn)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="relative w-full md:w-64">
              <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                type="search"
                placeholder="Search predictions..."
                className="pl-8"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>

            <Button asChild>
              <a href="/model-comparison">
                <GitCompare className="mr-2 h-4 w-4" />
                Compare Models
              </a>
            </Button>
          </div>

          <div className="flex gap-2">
            <Button variant="outline">
              <FileText className="mr-2 h-4 w-4" />
              Documentation
            </Button>
            <Button variant="outline">
              <Download className="mr-2 h-4 w-4" />
              Export
            </Button>
          </div>
        </div>

        <Tabs defaultValue="global" className="space-y-4">
          <TabsList>
            <TabsTrigger value="global">Global Explanations</TabsTrigger>
            <TabsTrigger value="local">Individual Predictions</TabsTrigger>
            <TabsTrigger value="advanced">Advanced Analysis</TabsTrigger>
          </TabsList>

          <TabsContent value="global" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <ShapSummaryPlot modelId={selectedModel} />
              <FeatureImportance modelId={selectedModel} />
            </div>

            <Card>
              <CardHeader>
                <CardTitle>Feature Interactions</CardTitle>
                <CardDescription>How features interact with each other to influence predictions</CardDescription>
              </CardHeader>
              <CardContent className="h-[400px] flex items-center justify-center">
                <p className="text-muted-foreground">Feature interaction visualization will appear here</p>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="local" className="space-y-6">
            <PredictionExplainer modelId={selectedModel} />
          </TabsContent>

          <TabsContent value="advanced" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Partial Dependence Plots</CardTitle>
                  <CardDescription>How predictions change when varying a single feature</CardDescription>
                </CardHeader>
                <CardContent className="h-[400px] flex items-center justify-center">
                  <p className="text-muted-foreground">Partial dependence plots will appear here</p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Accumulated Local Effects</CardTitle>
                  <CardDescription>Feature effects that account for correlations</CardDescription>
                </CardHeader>
                <CardContent className="h-[400px] flex items-center justify-center">
                  <p className="text-muted-foreground">ALE plots will appear here</p>
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>Permutation Feature Importance</CardTitle>
                <CardDescription>Model-agnostic feature importance based on performance impact</CardDescription>
              </CardHeader>
              <CardContent className="h-[400px] flex items-center justify-center">
                <p className="text-muted-foreground">Permutation importance visualization will appear here</p>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
