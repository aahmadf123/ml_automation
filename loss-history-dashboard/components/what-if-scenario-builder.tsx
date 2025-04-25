"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Slider } from "@/components/ui/slider"
import { Button } from "@/components/ui/button"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Label } from "@/components/ui/label"
import { Sparkles, Lightbulb, BarChart3, LineChart, RefreshCw, ArrowUpRight, ArrowDownRight } from "lucide-react"
import { motion } from "framer-motion"

type ScenarioParameter = {
  id: string
  name: string
  description: string
  currentValue: number
  minValue: number
  maxValue: number
  step: number
  unit: string
  impact: "high" | "medium" | "low"
}

type MetricImpact = {
  name: string
  currentValue: number
  projectedValue: number
  change: number
  trend: "up" | "down" | "neutral"
}

export function WhatIfScenarioBuilder() {
  const [activeScenario, setActiveScenario] = useState<string>("custom")
  const [isCalculating, setIsCalculating] = useState<boolean>(false)
  const [hasResults, setHasResults] = useState<boolean>(false)
  
  const [parameters, setParameters] = useState<ScenarioParameter[]>([
    {
      id: "learning_rate",
      name: "Learning Rate",
      description: "Step size at each iteration of the optimization algorithm",
      currentValue: 0.01,
      minValue: 0.001,
      maxValue: 0.1,
      step: 0.001,
      unit: "",
      impact: "high"
    },
    {
      id: "regularization",
      name: "Regularization",
      description: "Reduces overfitting by penalizing large coefficients",
      currentValue: 0.1,
      minValue: 0,
      maxValue: 1,
      step: 0.05,
      unit: "",
      impact: "medium"
    },
    {
      id: "batch_size",
      name: "Batch Size",
      description: "Number of samples processed before model update",
      currentValue: 64,
      minValue: 16,
      maxValue: 256,
      step: 16,
      unit: "samples",
      impact: "medium"
    },
    {
      id: "dropout_rate",
      name: "Dropout Rate",
      description: "Fraction of input units to drop during training",
      currentValue: 0.2,
      minValue: 0,
      maxValue: 0.5,
      step: 0.05,
      unit: "",
      impact: "low"
    }
  ])
  
  const [metrics, setMetrics] = useState<MetricImpact[]>([
    {
      name: "Accuracy",
      currentValue: 83.7,
      projectedValue: 83.7,
      change: 0,
      trend: "neutral"
    },
    {
      name: "Processing Time",
      currentValue: 156,
      projectedValue: 156,
      change: 0,
      trend: "neutral"
    },
    {
      name: "Memory Usage",
      currentValue: 1.8,
      projectedValue: 1.8,
      change: 0,
      trend: "neutral"
    },
    {
      name: "R² Score",
      currentValue: 0.79,
      projectedValue: 0.79,
      change: 0,
      trend: "neutral"
    }
  ])
  
  // Predefined scenarios
  const scenarios = {
    performance: {
      learning_rate: 0.005,
      regularization: 0.2,
      batch_size: 32,
      dropout_rate: 0.3
    },
    speed: {
      learning_rate: 0.03,
      regularization: 0.05,
      batch_size: 128,
      dropout_rate: 0.1
    },
    balanced: {
      learning_rate: 0.01,
      regularization: 0.15,
      batch_size: 64,
      dropout_rate: 0.2
    },
    custom: {}
  }
  
  // Handle scenario selection
  const handleScenarioChange = (scenarioId: string) => {
    setActiveScenario(scenarioId)
    
    if (scenarioId !== "custom") {
      const selectedScenario = scenarios[scenarioId as keyof typeof scenarios]
      
      // Update parameters based on selected scenario
      setParameters(prevParams => 
        prevParams.map(param => ({
          ...param,
          currentValue: selectedScenario[param.id as keyof typeof selectedScenario] || param.currentValue
        }))
      )
    }
  }
  
  // Handle parameter change
  const handleParameterChange = (parameterId: string, newValue: number) => {
    setParameters(prevParams => 
      prevParams.map(param => 
        param.id === parameterId ? { ...param, currentValue: newValue } : param
      )
    )
    
    // Set to custom scenario when user manually changes parameters
    setActiveScenario("custom")
  }
  
  // Calculate projected metrics based on parameter changes
  const calculateProjection = () => {
    setIsCalculating(true)
    
    // Simulate API call or complex calculation
    setTimeout(() => {
      // Get learning rate parameter
      const learningRate = parameters.find(p => p.id === "learning_rate")?.currentValue || 0.01
      const regularization = parameters.find(p => p.id === "regularization")?.currentValue || 0.1
      const batchSize = parameters.find(p => p.id === "batch_size")?.currentValue || 64
      
      // Simple simulation calculations (in a real app, these would be proper ML estimates)
      const accuracyChange = (0.01 - learningRate) * 10 + (regularization - 0.1) * 5 + Math.log(64 / batchSize) * 2
      const timeChange = (batchSize / 64 - 1) * 20 - (learningRate / 0.01) * 10
      const memoryChange = (batchSize / 64) * 15
      const r2Change = (0.01 - learningRate) * 8 + (regularization - 0.1) * 12
      
      // Update metrics with projected values
      setMetrics(prevMetrics => 
        prevMetrics.map(metric => {
          let projectedValue = metric.currentValue
          let change = 0
          let trend: "up" | "down" | "neutral" = "neutral"
          
          switch (metric.name) {
            case "Accuracy":
              projectedValue = Math.min(100, Math.max(50, metric.currentValue + accuracyChange))
              change = projectedValue - metric.currentValue
              trend = change > 0 ? "up" : change < 0 ? "down" : "neutral"
              break
            case "Processing Time":
              projectedValue = Math.max(50, metric.currentValue + timeChange)
              change = projectedValue - metric.currentValue
              trend = change < 0 ? "up" : change > 0 ? "down" : "neutral" // Inverse for time (less is better)
              break
            case "Memory Usage":
              projectedValue = Math.max(0.5, metric.currentValue + (memoryChange / 100))
              change = projectedValue - metric.currentValue
              trend = change < 0 ? "up" : change > 0 ? "down" : "neutral" // Inverse for memory (less is better)
              break
            case "R² Score":
              projectedValue = Math.min(1, Math.max(0, metric.currentValue + (r2Change / 100)))
              change = projectedValue - metric.currentValue
              trend = change > 0 ? "up" : change < 0 ? "down" : "neutral"
              break
          }
          
          return {
            ...metric,
            projectedValue: Number(projectedValue.toFixed(2)),
            change: Number(change.toFixed(2)),
            trend
          }
        })
      )
      
      setIsCalculating(false)
      setHasResults(true)
    }, 1500)
  }
  
  // Reset parameters to default
  const resetParameters = () => {
    setParameters(prevParams => 
      prevParams.map(param => {
        const defaultScenario = scenarios.balanced
        return {
          ...param,
          currentValue: defaultScenario[param.id as keyof typeof defaultScenario] || param.currentValue
        }
      })
    )
    setActiveScenario("balanced")
    setHasResults(false)
  }
  
  // Apply scenario to model (in a real app, this would trigger a model retraining)
  const applyScenario = () => {
    // This would be an API call in a real application
    alert("In a real application, this would apply the scenario and retrain the model.")
  }
  
  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Sparkles className="h-5 w-5 text-amber-500" />
          What-If Scenario Builder
        </CardTitle>
        <CardDescription>
          Adjust parameters and see the projected impact on model performance
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Scenario Selection */}
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <Label>Choose Scenario</Label>
            <Button 
              variant="ghost" 
              size="sm" 
              className="h-7 gap-1 text-xs"
              onClick={resetParameters}
            >
              <RefreshCw className="h-3 w-3" />
              Reset
            </Button>
          </div>
          
          <Tabs value={activeScenario} onValueChange={handleScenarioChange} className="w-full">
            <TabsList className="grid grid-cols-4 mb-4">
              <TabsTrigger value="performance">Performance</TabsTrigger>
              <TabsTrigger value="speed">Speed</TabsTrigger>
              <TabsTrigger value="balanced">Balanced</TabsTrigger>
              <TabsTrigger value="custom">Custom</TabsTrigger>
            </TabsList>
            
            <div className="relative rounded-md border p-3 mb-4">
              <p className="text-sm font-medium mb-1">Scenario Description</p>
              <p className="text-xs text-gray-500">
                {activeScenario === "performance" && "Optimized for maximum accuracy and performance, may require more processing time."}
                {activeScenario === "speed" && "Optimized for fast processing and inference, with slight trade-offs in accuracy."}
                {activeScenario === "balanced" && "A balanced approach between performance and speed."}
                {activeScenario === "custom" && "Custom parameter configuration."}
              </p>
            </div>
          </Tabs>
        </div>
        
        {/* Parameters */}
        <div className="space-y-4">
          <h3 className="text-sm font-medium">Model Parameters</h3>
          
          {parameters.map(param => (
            <div key={param.id} className="space-y-2">
              <div className="flex items-center justify-between">
                <div>
                  <Label className="flex items-center gap-1">
                    {param.name}
                    <span className={
                      param.impact === "high" ? "bg-red-100 text-red-800 text-[10px] px-1.5 py-0.5 rounded-full" :
                      param.impact === "medium" ? "bg-amber-100 text-amber-800 text-[10px] px-1.5 py-0.5 rounded-full" :
                      "bg-blue-100 text-blue-800 text-[10px] px-1.5 py-0.5 rounded-full"
                    }>
                      {param.impact} impact
                    </span>
                  </Label>
                  <p className="text-xs text-gray-500">{param.description}</p>
                </div>
                <div className="flex items-center text-sm font-medium">
                  {param.currentValue}
                  {param.unit && <span className="ml-1">{param.unit}</span>}
                </div>
              </div>
              
              <Slider
                value={[param.currentValue]}
                min={param.minValue}
                max={param.maxValue}
                step={param.step}
                onValueChange={(values) => handleParameterChange(param.id, values[0])}
                className="w-full"
              />
            </div>
          ))}
          
          <Button 
            onClick={calculateProjection}
            className="w-full mt-4"
            disabled={isCalculating}
          >
            {isCalculating ? (
              <>
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                Calculating...
              </>
            ) : (
              <>
                <Lightbulb className="h-4 w-4 mr-2" />
                Calculate Projected Impact
              </>
            )}
          </Button>
        </div>
        
        {/* Results */}
        {hasResults && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            className="space-y-4 mt-6"
          >
            <h3 className="text-sm font-medium flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              Projected Impact
            </h3>
            
            <div className="grid grid-cols-2 gap-4">
              {metrics.map(metric => (
                <Card key={metric.name} className="border">
                  <CardHeader className="pb-2 pt-4">
                    <CardTitle className="text-sm">{metric.name}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-end justify-between">
                      <div>
                        <div className="flex items-center">
                          <span className="text-2xl font-bold">
                            {metric.projectedValue}
                            {metric.name === "R² Score" ? "" : metric.name === "Memory Usage" ? "GB" : metric.name === "Processing Time" ? "ms" : "%"}
                          </span>
                          {metric.trend !== "neutral" && (
                            <span className={`ml-2 ${metric.trend === "up" ? "text-green-500" : "text-red-500"} flex items-center`}>
                              {metric.trend === "up" ? <ArrowUpRight className="h-4 w-4" /> : <ArrowDownRight className="h-4 w-4" />}
                              {metric.change > 0 ? "+" : ""}{metric.change}
                            </span>
                          )}
                        </div>
                        <p className="text-xs text-gray-500">
                          Current: {metric.currentValue}
                          {metric.name === "R² Score" ? "" : metric.name === "Memory Usage" ? "GB" : metric.name === "Processing Time" ? "ms" : "%"}
                        </p>
                      </div>
                      
                      <div className="h-12 w-16 flex items-end">
                        <LineChart className="h-full w-full text-blue-500 opacity-70" />
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
            
            <div className="flex justify-end gap-2 mt-4">
              <Button variant="outline" onClick={resetParameters}>
                Reset
              </Button>
              <Button onClick={applyScenario}>
                Apply Scenario
              </Button>
            </div>
          </motion.div>
        )}
      </CardContent>
    </Card>
  )
} 