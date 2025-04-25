"use client"

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import Image from "next/image"
import { Button } from "@/components/ui/button"
import Link from "next/link"
import { ArrowRight } from "lucide-react"

// Only keeping and updating models Model1 (Traditional) and Model4 (Enhanced/Fast Decay)
const modelData = [
  {
    id: "model1",
    name: "Traditional Model (48 Attributes)",
    version: "v1.2",
    metrics: {
      r2: 0.67,
      rmse: 0.572,
      mae: 0.431,
      accuracy: 0.84,
      precision: 0.82,
    },
    improvement: {
      r2: 0,
      rmse: 0,
      mae: 0,
      accuracy: 0,
      precision: 0,
    },
    features: [
      { name: "num_loss_3yr", importance: 24.3 },
      { name: "num_loss_yrs45", importance: 22.8 },
      { name: "num_loss_free_yrs", importance: 18.9 },
      { name: "policy_age", importance: 15.2 },
      { name: "construction_type", importance: 12.5 },
    ],
    type: "Regression",
    status: "baseline",
  },
  {
    id: "model4",
    name: "Enhanced Model (Fast Decay)",
    version: "v2.1",
    metrics: {
      r2: 0.79,
      rmse: 0.423,
      mae: 0.332,
      accuracy: 0.91,
      precision: 0.89,
    },
    improvement: {
      r2: 17.9,
      rmse: 26.0,
      mae: 22.9,
      accuracy: 8.3,
      precision: 8.5,
    },
    features: [
      { name: "lhdwc_5y_3d", importance: 31.2 },
      { name: "policy_age", importance: 18.7 },
      { name: "construction_type", importance: 14.3 },
      { name: "coverage_amount", importance: 12.8 },
      { name: "location_risk", importance: 11.7 },
    ],
    type: "Regression",
    status: "active",
  }
]

export function ModelComparisonShowcase() {
  const [activeModel, setActiveModel] = useState('model4')
  const [activeMetric, setActiveMetric] = useState('r2')
  
  const formatMetricName = (metricName: string) => {
    if (metricName === 'r2') return 'R² Score';
    if (metricName === 'rmse') return 'RMSE';
    if (metricName === 'mae') return 'MAE';
    return metricName.charAt(0).toUpperCase() + metricName.slice(1);
  }
  
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'bg-green-100 border-green-500 text-green-800';
      case 'baseline':
        return 'bg-blue-100 border-blue-500 text-blue-800';
      case 'failed':
        return 'bg-red-100 border-red-500 text-red-800';
      default:
        return 'bg-gray-100 border-gray-500 text-gray-800';
    }
  }
  
  return (
    <Card className="shadow-lg overflow-hidden border-t-4 border-t-blue-500">
      <CardHeader className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950 dark:to-indigo-950">
        <CardTitle className="flex items-center justify-center">
          <span className="text-2xl font-extrabold bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-600">
            R² Score Comparison
          </span>
        </CardTitle>
      </CardHeader>
      
      <CardContent className="p-6">
        <Tabs defaultValue="comparison" className="space-y-6">
          <TabsList className="grid grid-cols-2 w-full max-w-md mx-auto">
            <TabsTrigger value="comparison">Performance Chart</TabsTrigger>
            <TabsTrigger value="details">Model Details</TabsTrigger>
          </TabsList>
          
          <TabsContent value="comparison" className="space-y-6">
            <div className="flex justify-center space-x-4 mb-2">
              <Button 
                variant={activeMetric === 'r2' ? 'default' : 'outline'} 
                size="sm"
                onClick={() => setActiveMetric('r2')}
              >
                R² Score
              </Button>
              <Button 
                variant={activeMetric === 'rmse' ? 'default' : 'outline'} 
                size="sm"
                onClick={() => setActiveMetric('rmse')}
              >
                RMSE
              </Button>
              <Button 
                variant={activeMetric === 'mae' ? 'default' : 'outline'} 
                size="sm"
                onClick={() => setActiveMetric('mae')}
              >
                MAE
              </Button>
            </div>
            
            <div className="flex justify-center space-x-12 h-[300px] items-end">
              {modelData.map(model => (
                <div
                  key={model.id}
                  className={`relative flex flex-col items-center transition-all duration-300 ${activeModel === model.id ? 'scale-110' : 'opacity-70 hover:opacity-90'}`}
                  onMouseEnter={() => setActiveModel(model.id)}
                >
                  <div className="absolute -top-8 text-center w-full">
                    <Badge className={getStatusColor(model.status)}>
                      {model.status === 'baseline' ? 'TRADITIONAL' : 'ENHANCED'}
                    </Badge>
                  </div>
                  
                  <div 
                    className={`w-32 transition-height duration-500 rounded-t-md flex items-center justify-center
                      ${model.status === 'baseline' ? 'bg-blue-400 dark:bg-blue-700' : 
                        'bg-gradient-to-t from-green-400 to-teal-400 dark:from-green-600 dark:to-teal-600'}`}
                    style={{ 
                      height: `${(model.metrics[activeMetric as keyof typeof model.metrics] || 0) * 300}px`,
                    }}
                  >
                    <span className="text-white font-bold text-lg rotate-180" style={{ writingMode: 'vertical-rl' }}>
                      {activeMetric === 'r2' ? 
                        `${(model.metrics[activeMetric as keyof typeof model.metrics] * 100).toFixed(1)}%` : 
                        model.metrics[activeMetric as keyof typeof model.metrics].toFixed(3)}
                    </span>
                  </div>
                  
                  <p className="text-sm font-semibold mt-2 text-center max-w-[120px]">{model.name}</p>
                  
                  {model.id === 'model4' && activeMetric === 'r2' && (
                    <div className="absolute -right-4 top-1/2 transform -translate-y-1/2">
                      <Badge className="bg-green-100 border-green-500 text-green-800">
                        +17.9% better
                      </Badge>
                    </div>
                  )}
                </div>
              ))}
            </div>
            
            <div className="text-center text-sm text-gray-500 mt-8">
              {activeMetric === 'r2' ? (
                <p>Higher R² Score indicates better predictive accuracy</p>
              ) : activeMetric === 'rmse' ? (
                <p>Lower RMSE indicates better model fit</p>
              ) : (
                <p>Lower MAE indicates more accurate predictions</p>
              )}
            </div>
          </TabsContent>
          
          <TabsContent value="details">
            <Tabs defaultValue="model4">
              <TabsList className="grid grid-cols-2 w-full max-w-md mx-auto mb-6">
                <TabsTrigger value="model1">Traditional Model</TabsTrigger>
                <TabsTrigger value="model4">Enhanced Model</TabsTrigger>
              </TabsList>
              
              {modelData.map(model => (
                <TabsContent key={model.id} value={model.id} className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <div className="flex items-center space-x-2">
                        <h2 className="text-xl font-bold">{model.name}</h2>
                        <Badge className={getStatusColor(model.status)}>
                          {model.status === 'baseline' ? 'Traditional' : 'Enhanced'}
                        </Badge>
                      </div>
                      
                      <p className="text-sm text-gray-500 mt-1">
                        Version {model.version} • {model.type} Model
                      </p>
                      
                      <div className="mt-4 p-4 rounded-md bg-blue-50 dark:bg-blue-900">
                        <h3 className="font-medium text-blue-800 dark:text-blue-200">Description</h3>
                        <p className="text-sm text-blue-700 dark:text-blue-300 mt-1">
                          {model.id === 'model1' 
                            ? 'Traditional model using 48 original attributes to predict losses based on historical data.'
                            : 'Enhanced model using fast decay weighting to give higher importance to recent claims data.'}
                        </p>
                      </div>
                      
                      <h3 className="text-lg font-semibold mt-6 mb-2">Performance Metrics</h3>
                      <div className="space-y-2">
                        {Object.entries(model.metrics).map(([key, value]) => (
                          <div key={key} className="flex justify-between">
                            <span className="text-sm font-medium">{formatMetricName(key)}:</span>
                            <span className="text-sm">{key === 'r2' ? `${(value * 100).toFixed(1)}%` : value.toFixed(3)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                    
                    <div>
                      <h3 className="text-lg font-semibold mb-2">Top Features</h3>
                      <div className="space-y-3">
                        {model.features.map((feature, index) => (
                          <div key={index} className="space-y-1">
                            <div className="flex justify-between text-sm">
                              <span>{feature.name.replace(/_/g, ' ')}</span>
                              <span>{feature.importance.toFixed(1)}%</span>
                            </div>
                            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                              <div 
                                className={`${model.id === 'model1' ? 'bg-blue-500' : 'bg-green-500'} h-2 rounded-full`}
                                style={{ width: `${feature.importance}%` }}
                              ></div>
                            </div>
                          </div>
                        ))}
                      </div>
                      
                      {model.id === 'model4' && (
                        <div className="mt-6">
                          <h3 className="text-lg font-semibold mb-2">Improvement over Traditional Model</h3>
                          <div className="p-3 bg-green-50 dark:bg-green-900 rounded-md">
                            <p className="text-sm text-green-800 dark:text-green-200">
                              <strong>R² Score improved by 17.9%</strong> - This means significantly better predictions 
                              and more accurate risk assessment for pricing policies.
                            </p>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </TabsContent>
              ))}
            </Tabs>
          </TabsContent>
        </Tabs>
        
        <div className="mt-8 text-center">
          <Button variant="outline" asChild>
            <Link href="/model-comparison">
              <span>View Detailed Comparison</span>
              <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
        </div>
      </CardContent>
    </Card>
  )
} 