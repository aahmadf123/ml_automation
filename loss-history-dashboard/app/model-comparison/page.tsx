"use client";

import { DashboardHeader } from "@/components/dashboard-header";
import { DashboardSidebar } from "@/components/dashboard-sidebar";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { GitCompare, Download, BarChart, LineChart } from "lucide-react";

export default function ModelComparisonPage() {
  return (
    <div className="flex min-h-screen flex-col">
      <DashboardHeader title="Model Comparison" />
      <div className="container flex-1 items-start md:grid md:grid-cols-[220px_minmax(0,1fr)] md:gap-6 lg:grid-cols-[240px_minmax(0,1fr)] lg:gap-10">
        <DashboardSidebar />
        <main className="flex w-full flex-col overflow-hidden p-4 md:p-6">
          <Card>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Model Comparison Dashboard</CardTitle>
                  <CardDescription>
                    Compare models side-by-side with business impact
                  </CardDescription>
                </div>
                <div className="flex items-center gap-2">
                  <Button variant="outline" size="sm">
                    <Download className="h-4 w-4 mr-2" />
                    Export
                  </Button>
                  <Button size="sm">
                    <GitCompare className="h-4 w-4 mr-2" />
                    Compare Models
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="performance">
                <TabsList>
                  <TabsTrigger value="performance">Performance Metrics</TabsTrigger>
                  <TabsTrigger value="feature-importance">Feature Importance</TabsTrigger>
                  <TabsTrigger value="business-impact">Business Impact</TabsTrigger>
                </TabsList>
                
                <TabsContent value="performance" className="pt-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-lg">Model Performance Metrics</CardTitle>
                        <CardDescription>
                          Key metrics comparison across models
                        </CardDescription>
                      </CardHeader>
                      <CardContent className="pt-0">
                        <div className="h-80 flex items-center justify-center">
                          <p className="text-muted-foreground">Performance metrics chart will be displayed here</p>
                        </div>
                      </CardContent>
                    </Card>
                    
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-lg">Error Distribution</CardTitle>
                        <CardDescription>
                          Comparison of error distributions
                        </CardDescription>
                      </CardHeader>
                      <CardContent className="pt-0">
                        <div className="h-80 flex items-center justify-center">
                          <p className="text-muted-foreground">Error distribution chart will be displayed here</p>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </TabsContent>
                
                <TabsContent value="feature-importance" className="pt-4">
                  <Card>
                    <CardHeader>
                      <CardTitle>Feature Importance Comparison</CardTitle>
                      <CardDescription>
                        Compare feature importance across models
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="h-80 flex items-center justify-center">
                        <p className="text-muted-foreground">Feature importance comparison will be displayed here</p>
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>
                
                <TabsContent value="business-impact" className="pt-4">
                  <Card>
                    <CardHeader>
                      <CardTitle>Business Impact Analysis</CardTitle>
                      <CardDescription>
                        Compare business metrics and ROI
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="h-80 flex items-center justify-center">
                        <p className="text-muted-foreground">Business impact analysis will be displayed here</p>
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        </main>
      </div>
    </div>
  );
}

"use client"

import React from 'react'
import { useState, useEffect } from 'react'
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardHeader, 
  CardTitle 
} from '@/components/ui/card'
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from '@/components/ui/table'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { AlertCircle, Download, BarChart4, RefreshCw, BrainCircuit, TrendingUp, BarChart } from 'lucide-react'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import PageHeader from '@/components/ui/page-header'
import LoadingSpinner from '@/components/ui/loading-spinner'
import ComparisonTable, { ComparisonReportData } from '../../components/model-comparison/ComparisonTable'
import Link from 'next/link'

// Type definitions
type Metric = {
  name: string
  value: number
  unit?: string
  isHigherBetter: boolean
}

type ModelComparisonData = {
  modelId: string
  name: string
  metrics: Metric[]
  status: 'completed' | 'failed' | 'running'
  timestamp: string
  isBaseline?: boolean
  improvement?: Record<string, number>
}

type ComparisonReport = {
  id: string
  timestamp: string
  models: ModelComparisonData[]
  bestModel: string
  metricNames: string[]
  plots: { name: string, url: string }[]
}

const defaultMetrics = [
  'accuracy', 'precision', 'recall', 'f1_score', 
  'auc', 'mae', 'mse', 'rmse', 'r2'
]

export default async function ModelComparisonPage() {
  // Fetch model comparison reports
  const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || ''}/api/models/comparison`, {
    cache: 'no-store'
  });
  
  const reports: ComparisonReportData[] = await response.json();
  
  return (
    <div className="container mx-auto py-6">
      <div className="mb-8">
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2">Model Performance Comparison</h1>
            <p className="text-gray-600 max-w-3xl">
              Compare the traditional model using 48 attributes with our enhanced best-performing model.
              See how the R² score improvements translate to better business outcomes.
            </p>
          </div>
          
          <Link href="/model-comparison/ai-assistant">
            <Button className="gap-2">
              <BrainCircuit className="h-4 w-4" />
              AI Assistant
            </Button>
          </Link>
        </div>
        
        {/* Business Value Banner */}
        <Card className="mt-6 bg-gradient-to-r from-blue-50 to-indigo-50 border-blue-100">
          <CardContent className="p-6">
            <div className="flex flex-col md:flex-row items-center justify-between">
              <div className="mb-4 md:mb-0 md:mr-6">
                <h2 className="text-xl font-bold text-blue-900 mb-2">
                  Improved R² Score = Better Business Decisions
                </h2>
                <p className="text-blue-800">
                  Our enhanced model with fast decay weighting significantly outperforms the traditional 
                  approach with 48 old attributes, providing superior loss predictions.
                </p>
              </div>
              <div className="flex items-center gap-6 p-4 bg-white rounded-lg shadow-sm">
                <div className="text-center">
                  <BarChart className="h-8 w-8 text-blue-500 mx-auto mb-1" />
                  <div className="text-sm text-blue-700">Better Risk Assessment</div>
                </div>
                <div className="text-center">
                  <TrendingUp className="h-8 w-8 text-green-500 mx-auto mb-1" />
                  <div className="text-sm text-green-700">Improved Pricing Precision</div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
      
      <Tabs defaultValue={reports[0]?.id || "no-data"} className="w-full">
        <div className="flex items-center justify-between mb-4">
          <TabsList>
            {reports.map(report => (
              <TabsTrigger key={report.id} value={report.id}>
                Report {new Date(report.timestamp).toLocaleDateString()}
              </TabsTrigger>
            ))}
            {reports.length === 0 && (
              <TabsTrigger value="no-data">No Reports</TabsTrigger>
            )}
          </TabsList>
        </div>
        
        {reports.length === 0 ? (
          <Card>
            <CardHeader>
              <CardTitle>No Comparison Reports</CardTitle>
              <CardDescription>
                There are no model comparison reports available yet.
              </CardDescription>
            </CardHeader>
          </Card>
        ) : (
          reports.map(report => (
            <TabsContent key={report.id} value={report.id}>
              <ReportDetails report={report} />
            </TabsContent>
          ))
        )}
      </Tabs>
    </div>
  );
}

function ReportDetails({ report }: { report: ComparisonReportData }) {
  // Filter to only show models 1 and 4 if available
  const filteredModels = report.models.filter(model => 
    model.modelId === 'model1' || model.modelId === 'model4'
  );
  
  // Create a filtered report to pass to ComparisonTable
  const filteredReport = {
    ...report,
    models: filteredModels.length > 0 ? filteredModels : report.models
  };
  
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Model Performance Comparison</CardTitle>
          <CardDescription>
            Comparing traditional vs. enhanced model from {new Date(report.timestamp).toLocaleString()}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ComparisonTable report={filteredReport} primaryMetric="r2" />
        </CardContent>
      </Card>
      
      {report.plots && report.plots.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Performance Visualizations</CardTitle>
            <CardDescription>
              Visual representation of model performance metrics
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {report.plots.map(plot => (
                <div key={plot.name} className="border rounded-lg p-4">
                  <h3 className="text-lg font-medium mb-2">{plot.name.replace(/_/g, ' ')}</h3>
                  <div className="aspect-video bg-gray-100 flex items-center justify-center">
                    <img 
                      src={plot.url} 
                      alt={plot.name} 
                      className="max-w-full max-h-full object-contain"
                    />
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Business Impact Analysis</CardTitle>
          <CardDescription>
            How improved model performance translates to business outcomes
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="p-4 border rounded-lg bg-green-50 border-green-100">
              <h3 className="font-semibold text-green-800 mb-2">Pricing Accuracy</h3>
              <p className="text-green-700 text-sm">
                Improved R² score leads to more precise risk assessments, allowing for better pricing accuracy.
              </p>
            </div>
            
            <div className="p-4 border rounded-lg bg-blue-50 border-blue-100">
              <h3 className="font-semibold text-blue-800 mb-2">Claims Prediction</h3>
              <p className="text-blue-700 text-sm">
                The enhanced Fast Decay model better captures recent claims patterns, improving loss predictions.
              </p>
            </div>
            
            <div className="p-4 border rounded-lg bg-purple-50 border-purple-100">
              <h3 className="font-semibold text-purple-800 mb-2">Competitive Advantage</h3>
              <p className="text-purple-700 text-sm">
                Better predictions allow for more competitive pricing for lower-risk policies.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="flex justify-end mt-4">
        <Link href="/model-comparison/ai-assistant">
          <Button variant="outline" className="gap-2">
            <BrainCircuit className="h-4 w-4" />
            Ask AI Assistant about these results
          </Button>
        </Link>
      </div>
    </div>
  );
}
