"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { RefreshCw } from "lucide-react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

export function ModelPerformanceSummary() {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <div>
          <CardTitle>Model Performance Summary</CardTitle>
          <CardDescription>Latest metrics from MLflow for all models</CardDescription>
        </div>
        <Button variant="outline" size="sm">
          <RefreshCw className="mr-2 h-4 w-4" />
          Refresh
        </Button>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="model1">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="model1">model1</TabsTrigger>
            <TabsTrigger value="model2">model2</TabsTrigger>
            <TabsTrigger value="model3">model3</TabsTrigger>
            <TabsTrigger value="model4">model4</TabsTrigger>
            <TabsTrigger value="model5">model5</TabsTrigger>
          </TabsList>
          <TabsContent value="model1" className="space-y-4">
            <div className="mt-4 space-y-2">
              <div className="flex items-center justify-between">
                <div className="text-sm font-medium">Production (v5)</div>
                <div className="text-sm text-muted-foreground">Gradient Boosting</div>
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="text-sm">Feature Importance</div>
                </div>
                <div className="space-y-1">
                  <div className="flex items-center justify-between">
                    <div className="text-xs">ClaimAmount</div>
                    <div className="text-xs">32.0%</div>
                  </div>
                  <div className="h-2 w-full rounded-full bg-muted">
                    <div className="h-2 rounded-full bg-emerald-500" style={{ width: "32%" }}></div>
                  </div>
                </div>

                <div className="space-y-1">
                  <div className="flex items-center justify-between">
                    <div className="text-xs">PropertyAge</div>
                    <div className="text-xs">24.5%</div>
                  </div>
                  <div className="h-2 w-full rounded-full bg-muted">
                    <div className="h-2 rounded-full bg-emerald-500" style={{ width: "24.5%" }}></div>
                  </div>
                </div>

                <div className="space-y-1">
                  <div className="flex items-center justify-between">
                    <div className="text-xs">ClaimType</div>
                    <div className="text-xs">18.3%</div>
                  </div>
                  <div className="h-2 w-full rounded-full bg-muted">
                    <div className="h-2 rounded-full bg-emerald-500" style={{ width: "18.3%" }}></div>
                  </div>
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="model2">
            <div className="mt-4 text-sm text-muted-foreground">Model 2 performance metrics will appear here.</div>
          </TabsContent>

          <TabsContent value="model3">
            <div className="mt-4 text-sm text-muted-foreground">Model 3 performance metrics will appear here.</div>
          </TabsContent>

          <TabsContent value="model4">
            <div className="mt-4 text-sm text-muted-foreground">Model 4 performance metrics will appear here.</div>
          </TabsContent>

          <TabsContent value="model5">
            <div className="mt-4 text-sm text-muted-foreground">Model 5 performance metrics will appear here.</div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
