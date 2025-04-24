"use client"

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import Image from "next/image"

// Sample model comparison data
const modelData = [
  {
    id: "neural_network",
    name: "Neural Network",
    version: "v1.5",
    metrics: {
      accuracy: 0.942,
      precision: 0.935,
      recall: 0.927,
      f1: 0.931,
      auc: 0.95,
    },
    improvement: {
      accuracy: 7.2,
      precision: 8.0,
      recall: 7.6,
      f1: 7.8,
      auc: 4.4,
    },
    features: [
      { name: "property_age", importance: 24.5 },
      { name: "claim_amount", importance: 32.0 },
      { name: "claim_type", importance: 18.3 },
    ],
    type: "Classification",
    status: "deployed",
  },
  {
    id: "random_forest",
    name: "Random Forest",
    version: "v2.1",
    metrics: {
      accuracy: 0.915,
      precision: 0.908,
      recall: 0.912,
      f1: 0.91,
      auc: 0.92,
    },
    improvement: {
      accuracy: 4.5,
      precision: 4.9,
      recall: 3.9,
      f1: 4.2,
      auc: 2.2,
    },
    features: [
      { name: "property_age", importance: 28.1 },
      { name: "claim_amount", importance: 25.3 },
      { name: "claim_type", importance: 20.5 },
    ],
    type: "Classification",
    status: "deployed",
  },
  {
    id: "xgboost",
    name: "XGBoost",
    version: "v1.2",
    metrics: {
      accuracy: 0.87,
      precision: 0.857,
      recall: 0.871,
      f1: 0.863,
      auc: 0.91,
    },
    improvement: {
      accuracy: 0,
      precision: 0,
      recall: 0,
      f1: 0,
      auc: 0,
    },
    features: [
      { name: "property_age", importance: 22.7 },
      { name: "claim_amount", importance: 29.8 },
      { name: "claim_type", importance: 15.2 },
    ],
    type: "Classification",
    status: "baseline",
  },
  {
    id: "lightgbm",
    name: "LightGBM",
    version: "v0.8",
    metrics: {
      accuracy: 0,
      precision: 0,
      recall: 0,
      f1: 0,
      auc: 0,
    },
    improvement: {
      accuracy: 0,
      precision: 0,
      recall: 0,
      f1: 0,
      auc: 0,
    },
    features: [],
    type: "Classification",
    status: "failed",
  }
]

export function ModelComparisonShowcase() {
  const [activeModel, setActiveModel] = useState('neural_network')
  const [activeMetric, setActiveMetric] = useState('f1')
  
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'deployed':
        return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300'
      case 'baseline':
        return 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300'
      case 'failed':
        return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300'
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-300'
    }
  }
  
  const getImprovementColor = (value: number) => {
    if (value > 5) return 'text-green-600 dark:text-green-400'
    if (value > 0) return 'text-blue-600 dark:text-blue-400'
    return 'text-gray-600 dark:text-gray-400'
  }
  
  const formatMetricName = (name: string) => {
    switch(name) {
      case 'f1': return 'F1 Score'
      case 'auc': return 'AUC'
      default: return name.charAt(0).toUpperCase() + name.slice(1)
    }
  }
  
  return (
    <Card className="shadow-lg overflow-hidden border-t-4 border-t-purple-500">
      <CardHeader className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-950 dark:to-indigo-950">
        <CardTitle className="flex items-center justify-center">
          <span className="text-2xl font-extrabold bg-clip-text text-transparent bg-gradient-to-r from-purple-600 to-indigo-600">
            Model Performance Comparison
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
            <div className="flex flex-wrap gap-4 justify-center mb-4">
              <div className="flex items-center gap-2">
                <Badge variant="outline" className="bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300">Deployed</Badge>
                <Badge variant="outline" className="bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300">Baseline</Badge>
                <Badge variant="outline" className="bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300">Failed</Badge>
              </div>
            </div>
            
            <div className="h-[300px] relative mx-auto">
              <div className="flex items-end h-full justify-around px-8 pt-10 pb-4 mb-6">
                {modelData.map(model => (
                  <div
                    key={model.id}
                    className={`relative flex flex-col items-center transition-all duration-300 ${activeModel === model.id ? 'scale-110' : 'opacity-60 hover:opacity-80'}`}
                    onMouseEnter={() => setActiveModel(model.id)}
                  >
                    <div className="absolute -top-8 text-center w-full">
                      <Badge className={getStatusColor(model.status)}>
                        {model.status.toUpperCase()}
                      </Badge>
                    </div>
                    
                    <div 
                      className={`w-20 transition-height duration-500 rounded-t-md flex items-center justify-center
                        ${model.status === 'failed' ? 'bg-red-200 dark:bg-red-800' : 
                          (model.status === 'baseline' ? 'bg-blue-200 dark:bg-blue-800' : 
                            'bg-gradient-to-t from-indigo-400 to-purple-400 dark:from-indigo-600 dark:to-purple-600')}`}
                      style={{ 
                        height: model.status === 'failed' ? '30px' : `${(model.metrics[activeMetric as keyof typeof model.metrics] || 0) * 250}px`,
                      }}
                    >
                      {model.status !== 'failed' && (
                        <span className="text-white font-bold text-sm rotate-180" style={{ writingMode: 'vertical-rl' }}>
                          {(model.metrics[activeMetric as keyof typeof model.metrics] * 100).toFixed(1)}%
                        </span>
                      )}
                    </div>
                    
                    <p className="text-xs font-semibold mt-2 whitespace-nowrap">{model.name}</p>
                    
                    {model.status !== 'failed' && model.status !== 'baseline' && (
                      <p className={`text-xs font-medium mt-1 ${getImprovementColor(model.improvement[activeMetric as keyof typeof model.improvement])}`}>
                        +{model.improvement[activeMetric as keyof typeof model.improvement].toFixed(1)}%
                      </p>
                    )}
                  </div>
                ))}
                
                {/* Y-axis */}
                <div className="absolute left-0 top-0 h-full flex flex-col justify-between">
                  <div className="text-xs text-gray-500">100%</div>
                  <div className="text-xs text-gray-500">75%</div>
                  <div className="text-xs text-gray-500">50%</div>
                  <div className="text-xs text-gray-500">25%</div>
                  <div className="text-xs text-gray-500">0%</div>
                </div>
              </div>
              
              <div className="flex justify-center gap-2 mt-6">
                {['accuracy', 'precision', 'recall', 'f1', 'auc'].map(metric => (
                  <Badge
                    key={metric}
                    variant={activeMetric === metric ? 'default' : 'outline'}
                    className={`cursor-pointer ${activeMetric === metric ? 'bg-purple-500 hover:bg-purple-600' : ''}`}
                    onClick={() => setActiveMetric(metric)}
                  >
                    {formatMetricName(metric)}
                  </Badge>
                ))}
              </div>
            </div>
          </TabsContent>
          
          <TabsContent value="details">
            <Tabs defaultValue={activeModel} className="space-y-4">
              <TabsList className="grid grid-cols-4 mx-auto w-full max-w-2xl">
                {modelData.map(model => (
                  <TabsTrigger key={model.id} value={model.id} className="text-xs">
                    {model.name}
                  </TabsTrigger>
                ))}
              </TabsList>
              
              {modelData.map(model => (
                <TabsContent key={model.id} value={model.id} className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h3 className="text-lg font-semibold mb-2">Model Details</h3>
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-sm font-medium">Model Type:</span>
                          <span className="text-sm">{model.type}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm font-medium">Version:</span>
                          <span className="text-sm">{model.version}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm font-medium">Status:</span>
                          <Badge className={getStatusColor(model.status)}>
                            {model.status.toUpperCase()}
                          </Badge>
                        </div>
                      </div>
                      
                      {model.status !== 'failed' && (
                        <>
                          <h3 className="text-lg font-semibold mt-6 mb-2">Performance Metrics</h3>
                          <div className="space-y-2">
                            {Object.entries(model.metrics).map(([key, value]) => (
                              <div key={key} className="flex justify-between">
                                <span className="text-sm font-medium">{formatMetricName(key)}:</span>
                                <span className="text-sm">{(value * 100).toFixed(1)}%</span>
                              </div>
                            ))}
                          </div>
                        </>
                      )}
                    </div>
                    
                    {model.status !== 'failed' && (
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
                                  className="bg-purple-500 h-2 rounded-full" 
                                  style={{ width: `${feature.importance}%` }}
                                ></div>
                              </div>
                            </div>
                          ))}
                        </div>
                        
                        {model.status !== 'baseline' && (
                          <div className="mt-6">
                            <h3 className="text-lg font-semibold mb-2">Improvement over Baseline</h3>
                            <div className="p-3 bg-green-50 dark:bg-green-900 rounded-md">
                              <p className="text-sm text-green-800 dark:text-green-200">
                                This model shows an average improvement of {
                                  (Object.values(model.improvement).reduce((a, b) => a + b, 0) / Object.values(model.improvement).length).toFixed(1)
                                }% across all metrics compared to the baseline model.
                              </p>
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                    
                    {model.status === 'failed' && (
                      <div className="p-4 bg-red-50 dark:bg-red-900 rounded-md">
                        <h3 className="text-lg font-semibold text-red-800 dark:text-red-200 mb-2">Training Failed</h3>
                        <p className="text-sm text-red-700 dark:text-red-300">
                          This model failed during training due to convergence issues. The training process was unable 
                          to optimize the model parameters properly. Consider adjusting hyperparameters or reviewing 
                          the preprocessing steps.
                        </p>
                      </div>
                    )}
                  </div>
                </TabsContent>
              ))}
            </Tabs>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
} 