"use client"

import { DashboardHeader } from "@/components/dashboard-header"
import { PageHeader } from "@/components/page-header"
import { ModelExplanationComparison } from "@/components/model-explanation-comparison"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Download, RefreshCw, FileText, GitCompare } from "lucide-react"

export default function ModelExplanationComparisonPage() {
  return (
    <div className="flex flex-col min-h-screen">
      <DashboardHeader />
      <div className="flex-1 p-6 space-y-6 md:ml-64">
        <PageHeader
          title="Model Explanation Comparison"
          description="Compare explanations across different model versions to understand how models evolve"
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

        <Tabs defaultValue="version-comparison" className="space-y-4">
          <TabsList>
            <TabsTrigger value="version-comparison">Version Comparison</TabsTrigger>
            <TabsTrigger value="explanation-history">Explanation History</TabsTrigger>
            <TabsTrigger value="model-lineage">Model Lineage</TabsTrigger>
          </TabsList>

          <TabsContent value="version-comparison">
            <ModelExplanationComparison />
          </TabsContent>

          <TabsContent value="explanation-history">
            <Card>
              <CardHeader>
                <CardTitle>Explanation History</CardTitle>
                <CardDescription>
                  Track how explanations for specific predictions have evolved over time
                </CardDescription>
              </CardHeader>
              <CardContent className="h-[600px] flex items-center justify-center">
                <div className="text-center">
                  <p className="text-muted-foreground mb-4">This feature is coming soon</p>
                  <Button variant="outline">
                    <FileText className="mr-2 h-4 w-4" />
                    Request Early Access
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="model-lineage">
            <Card>
              <CardHeader>
                <CardTitle>Model Lineage</CardTitle>
                <CardDescription>Visualize the evolution of models and their explanations over time</CardDescription>
              </CardHeader>
              <CardContent className="h-[600px] flex items-center justify-center">
                <div className="text-center">
                  <p className="text-muted-foreground mb-4">This feature is coming soon</p>
                  <Button variant="outline">
                    <GitCompare className="mr-2 h-4 w-4" />
                    Request Early Access
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
