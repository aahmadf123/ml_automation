"use client"

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { ArrowRight, LineChart, BarChart2, PieChart, TrendingUp } from "lucide-react"
import { cn } from "@/lib/utils"
import { Skeleton } from "@/components/ui/skeleton"
import Link from 'next/link'

type ModelResult = {
  id: string
  name: string
  version: string
  accuracy: number
  precision: number
  recall: number
  f1Score: number
  improvementPercent: number
  trainingDate: string
}

interface ComparisonShowcaseProps {
  className?: string
  limit?: number
  showViewAll?: boolean
  loading?: boolean
}

export function ComparisonShowcase({
  className,
  limit = 3,
  showViewAll = true,
  loading = false
}: ComparisonShowcaseProps) {
  const [selectedMetric, setSelectedMetric] = useState<string>("accuracy")
  const [models, setModels] = useState<ModelResult[]>([])
  
  // Simulated data - in a real app, this would come from an API call
  useEffect(() => {
    // Mock data
    const mockModels: ModelResult[] = [
      {
        id: "model-1",
        name: "Random Forest",
        version: "v2.3",
        accuracy: 0.92,
        precision: 0.89,
        recall: 0.88,
        f1Score: 0.885,
        improvementPercent: 5.2,
        trainingDate: "2023-10-15"
      },
      {
        id: "model-2",
        name: "XGBoost",
        version: "v1.8",
        accuracy: 0.95,
        precision: 0.94,
        recall: 0.93,
        f1Score: 0.935,
        improvementPercent: 7.8,
        trainingDate: "2023-09-28"
      },
      {
        id: "model-3",
        name: "Neural Network",
        version: "v4.1",
        accuracy: 0.96,
        precision: 0.95,
        recall: 0.94,
        f1Score: 0.945,
        improvementPercent: 3.1,
        trainingDate: "2023-11-05"
      },
      {
        id: "model-4",
        name: "Gradient Boosting",
        version: "v2.0",
        accuracy: 0.91,
        precision: 0.90,
        recall: 0.87,
        f1Score: 0.884,
        improvementPercent: 2.3,
        trainingDate: "2023-10-30"
      },
    ]
    
    // Sort by improvement percentage
    const sortedModels = [...mockModels].sort((a, b) => b.improvementPercent - a.improvementPercent)
    setModels(sortedModels.slice(0, limit))
  }, [limit])

  const getMetricValue = (model: ModelResult, metricName: string) => {
    switch (metricName) {
      case "accuracy":
        return model.accuracy
      case "precision":
        return model.precision
      case "recall":
        return model.recall
      case "f1Score":
        return model.f1Score
      default:
        return model.accuracy
    }
  }

  const getMetricIcon = (metricName: string) => {
    switch (metricName) {
      case "accuracy":
        return <LineChart className="h-4 w-4" />
      case "precision":
        return <BarChart2 className="h-4 w-4" />
      case "recall":
        return <PieChart className="h-4 w-4" />
      case "f1Score":
        return <TrendingUp className="h-4 w-4" />
      default:
        return <LineChart className="h-4 w-4" />
    }
  }

  if (loading) {
    return (
      <Card className={cn("", className)}>
        <CardHeader>
          <Skeleton className="h-8 w-3/4 mb-2" />
          <Skeleton className="h-4 w-1/2" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-10 w-full mb-4" />
          {[1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-24 w-full mb-3" />
          ))}
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className={cn("", className)}>
      <CardHeader>
        <CardTitle className="text-xl">Model Performance Showcase</CardTitle>
        <CardDescription>
          Compare the latest improvements across different models
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <Tabs defaultValue="accuracy" value={selectedMetric} onValueChange={setSelectedMetric}>
          <TabsList className="grid grid-cols-4 mb-4">
            <TabsTrigger value="accuracy" className="flex items-center gap-1">
              <LineChart className="h-4 w-4" />
              <span className="hidden sm:inline">Accuracy</span>
            </TabsTrigger>
            <TabsTrigger value="precision" className="flex items-center gap-1">
              <BarChart2 className="h-4 w-4" />
              <span className="hidden sm:inline">Precision</span>
            </TabsTrigger>
            <TabsTrigger value="recall" className="flex items-center gap-1">
              <PieChart className="h-4 w-4" />
              <span className="hidden sm:inline">Recall</span>
            </TabsTrigger>
            <TabsTrigger value="f1Score" className="flex items-center gap-1">
              <TrendingUp className="h-4 w-4" />
              <span className="hidden sm:inline">F1 Score</span>
            </TabsTrigger>
          </TabsList>
          
          {["accuracy", "precision", "recall", "f1Score"].map((metric) => (
            <TabsContent key={metric} value={metric} className="space-y-4 mt-0">
              {models.map((model) => (
                <div 
                  key={model.id} 
                  className="relative overflow-hidden border rounded-lg p-4 transition-all hover:shadow-md"
                >
                  <div className="absolute top-0 left-0 h-full w-1 bg-gradient-to-b from-blue-500 to-purple-600"></div>
                  <div className="grid grid-cols-12 gap-4 items-center">
                    <div className="col-span-6 md:col-span-4">
                      <h3 className="font-semibold text-sm md:text-base">{model.name}</h3>
                      <p className="text-xs text-muted-foreground">{model.version} - {new Date(model.trainingDate).toLocaleDateString()}</p>
                    </div>
                    <div className="col-span-3 md:col-span-3 text-center">
                      <div className="flex items-center justify-center mb-1">
                        {getMetricIcon(metric)}
                        <span className="text-xs ml-1">{metric === "f1Score" ? "F1" : metric}</span>
                      </div>
                      <p className="text-base md:text-lg font-bold">{(getMetricValue(model, metric) * 100).toFixed(1)}%</p>
                    </div>
                    <div className="col-span-3 md:col-span-3 text-center">
                      <div className="text-xs text-muted-foreground mb-1">Improvement</div>
                      <p className="text-base md:text-lg font-bold text-green-600">+{model.improvementPercent.toFixed(1)}%</p>
                    </div>
                    <div className="hidden md:block md:col-span-2 text-right">
                      <Button variant="ghost" size="sm" asChild>
                        <Link href={`/model-comparison/${model.id}`}>
                          <span>Details</span>
                          <ArrowRight className="ml-1 h-3 w-3" />
                        </Link>
                      </Button>
                    </div>
                  </div>
                </div>
              ))}
            </TabsContent>
          ))}
        </Tabs>
      </CardContent>
      {showViewAll && (
        <CardFooter className="flex justify-center border-t pt-4">
          <Button variant="outline" asChild>
            <Link href="/model-comparison">
              <span>View All Comparisons</span>
              <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
        </CardFooter>
      )}
    </Card>
  )
} 