"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useState } from "react"
import { AlertTriangle, Sliders } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useToast } from "@/hooks/use-toast"

// Mock data for model performance
const modelData = {
  model1: {
    rmse: { value: 0.42, change: -0.05, threshold: 0.5, warning: false },
    mse: { value: 0.18, change: -0.03, threshold: 0.25, warning: false },
    mae: { value: 0.36, change: -0.02, threshold: 0.4, warning: false },
    r2: { value: 0.87, change: 0.03, threshold: 0.8, warning: false },
  },
  model2: {
    rmse: { value: 0.38, change: -0.02, threshold: 0.5, warning: false },
    mse: { value: 0.14, change: -0.01, threshold: 0.25, warning: false },
    mae: { value: 0.32, change: -0.03, threshold: 0.4, warning: false },
    r2: { value: 0.89, change: 0.01, threshold: 0.8, warning: false },
  },
  model3: {
    rmse: { value: 0.55, change: 0.02, threshold: 0.5, warning: true },
    mse: { value: 0.3, change: 0.01, threshold: 0.25, warning: true },
    mae: { value: 0.39, change: 0.03, threshold: 0.4, warning: false },
    r2: { value: 0.78, change: -0.02, threshold: 0.8, warning: true },
  },
}

export function ModelPerformance() {
  const { toast } = useToast()
  const [activeModel, setActiveModel] = useState("model1")

  const handleManageThresholds = () => {
    // In a real app, this would navigate to the threshold management page
    // For now, we'll just show a toast
    toast({
      title: "Manage Thresholds",
      description: "Navigating to threshold management page...",
    })
    // In a real app: router.push("/threshold-management")
    window.location.href = "/threshold-management"
  }

  const hasWarnings = (modelKey: string) => {
    const model = modelData[modelKey as keyof typeof modelData]
    return Object.values(model).some((metric) => metric.warning)
  }

  return (
    <Card className="shadow-md">
      <CardHeader className="pb-2">
        <div className="flex justify-between items-center">
          <div>
            <CardTitle>Model Performance</CardTitle>
            <CardDescription>Key metrics compared to thresholds</CardDescription>
          </div>
          <Button variant="outline" size="sm" onClick={handleManageThresholds}>
            <Sliders className="h-4 w-4 mr-2" />
            Manage Thresholds
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="model1" onValueChange={setActiveModel}>
          <TabsList className="grid grid-cols-3 mb-4">
            <TabsTrigger value="model1">
              Model 1{hasWarnings("model1") && <AlertTriangle className="h-3 w-3 ml-1 text-yellow-500" />}
            </TabsTrigger>
            <TabsTrigger value="model2">
              Model 2{hasWarnings("model2") && <AlertTriangle className="h-3 w-3 ml-1 text-yellow-500" />}
            </TabsTrigger>
            <TabsTrigger value="model3">
              Model 3{hasWarnings("model3") && <AlertTriangle className="h-3 w-3 ml-1 text-yellow-500" />}
            </TabsTrigger>
          </TabsList>

          {Object.keys(modelData).map((modelKey) => (
            <TabsContent key={modelKey} value={modelKey} className="space-y-4">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {Object.entries(modelData[modelKey as keyof typeof modelData]).map(([metric, data]) => {
                  const isGood = metric === "r2" ? data.value >= data.threshold : data.value <= data.threshold
                  return (
                    <Card
                      key={metric}
                      className={`overflow-hidden ${data.warning ? "border-red-400 dark:border-red-500" : ""}`}
                    >
                      <CardContent className="p-4">
                        <div className="flex justify-between items-center">
                          <h3 className="font-medium text-sm uppercase">{metric}</h3>
                          {data.warning && <AlertTriangle className="h-4 w-4 text-red-500" />}
                        </div>
                        <div className="flex items-center mt-1">
                          <span className={`text-xl font-bold ${isGood ? "text-green-500" : "text-red-500"}`}>
                            {data.value.toFixed(2)}
                          </span>
                          <span
                            className={`text-sm ml-2 ${
                              (metric === "r2" ? data.change > 0 : data.change < 0) ? "text-green-500" : "text-red-500"
                            }`}
                          >
                            {data.change > 0 ? "+" : ""}
                            {data.change.toFixed(2)}
                          </span>
                        </div>
                        <div className="text-xs text-muted-foreground mt-1">Threshold: {data.threshold.toFixed(2)}</div>
                      </CardContent>
                    </Card>
                  )
                })}
              </div>

              {hasWarnings(modelKey) && (
                <div className="p-3 rounded-md border border-yellow-200 bg-yellow-50 dark:bg-yellow-900/20 dark:border-yellow-800">
                  <div className="flex items-start space-x-3">
                    <AlertTriangle className="h-5 w-5 text-yellow-500 mt-0.5" />
                    <div>
                      <h3 className="font-medium text-yellow-800 dark:text-yellow-300">
                        Performance Thresholds Exceeded
                      </h3>
                      <p className="mt-1 text-sm text-yellow-700 dark:text-yellow-400">
                        Some metrics for {modelKey.replace(/([A-Z])/g, " $1").trim()} are outside of acceptable
                        thresholds. Consider retraining or adjusting thresholds.
                      </p>
                      <div className="mt-2 flex space-x-2">
                        <Button
                          size="sm"
                          onClick={() => {
                            toast({
                              title: "Retraining Initiated",
                              description: `Retraining process started for ${modelKey}`,
                            })
                          }}
                        >
                          Retrain Model
                        </Button>
                        <Button size="sm" variant="outline" onClick={handleManageThresholds}>
                          Adjust Thresholds
                        </Button>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <div className="mt-4 aspect-video w-full bg-muted rounded-lg flex items-center justify-center">
                <p className="text-muted-foreground">RMSE over last N runs chart for {activeModel}</p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                <div className="aspect-square bg-muted rounded-lg flex items-center justify-center">
                  <p className="text-muted-foreground">Actual vs. Predicted scatter</p>
                </div>
                <div className="aspect-square bg-muted rounded-lg flex items-center justify-center">
                  <p className="text-muted-foreground">SHAP summary</p>
                </div>
              </div>

              <div className="flex justify-end space-x-2 mt-4">
                <Button variant="outline">Download Data</Button>
                <Button>Export Images</Button>
              </div>
            </TabsContent>
          ))}
        </Tabs>
      </CardContent>
    </Card>
  )
}
