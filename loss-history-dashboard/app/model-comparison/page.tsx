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
import { AlertCircle, Download, BarChart4, RefreshCw } from 'lucide-react'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import PageHeader from '@/components/ui/page-header'
import LoadingSpinner from '@/components/ui/loading-spinner'
import ComparisonTable, { ComparisonReportData } from '../../components/model-comparison/ComparisonTable'

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
      <h1 className="text-3xl font-bold mb-6">Model Comparison</h1>
      
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
          
          {reports.length > 0 && (
            <MetricSelector 
              metrics={reports[0].metricNames} 
              defaultMetric="f1_score"
            />
          )}
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

function MetricSelector({ metrics, defaultMetric }: { metrics: string[], defaultMetric: string }) {
  // In a real implementation, this would update state and affect the table
  return (
    <div className="flex items-center space-x-2">
      <label className="text-sm font-medium">Primary Metric:</label>
      <Select defaultValue={metrics.includes(defaultMetric) ? defaultMetric : metrics[0]}>
        <SelectTrigger className="w-[180px]">
          <SelectValue placeholder="Select metric" />
        </SelectTrigger>
        <SelectContent>
          {metrics.map(metric => (
            <SelectItem key={metric} value={metric}>
              {metric.replace(/_/g, ' ')}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}

function ReportDetails({ report }: { report: ComparisonReportData }) {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Models Comparison</CardTitle>
          <CardDescription>
            Comparing {report.models.length} models from {new Date(report.timestamp).toLocaleString()}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ComparisonTable report={report} primaryMetric="f1_score" />
        </CardContent>
      </Card>
      
      {report.plots.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Performance Comparison Plots</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {report.plots.map(plot => (
                <div key={plot.name} className="border rounded-lg p-4">
                  <h3 className="text-lg font-medium mb-2">{plot.name}</h3>
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
    </div>
  );
}
