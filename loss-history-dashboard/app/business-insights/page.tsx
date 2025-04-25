"use client";

import { useState } from "react";
import { DashboardHeader } from "@/components/dashboard-header";
import { DashboardSidebar } from "@/components/dashboard-sidebar";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { 
  TrendingUp, 
  DollarSign, 
  RefreshCw, 
  Download, 
  BarChart, 
  PieChart, 
  LineChart,
  Zap,
  ArrowUpRight,
  ArrowDownRight,
  Calculator,
  Lightbulb
} from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

export default function BusinessInsightsPage() {
  const [timeRange, setTimeRange] = useState("ytd");
  const [modelType, setModelType] = useState("enhanced");
  
  // Helper functions to get business metrics - these would be replaced with actual data in a real implementation
  const getROI = () => {
    return {
      value: 347,
      trend: "+12.4%",
      isPositive: true
    };
  };
  
  const getCostSavings = () => {
    return {
      value: 1240000,
      trend: "+8.7%",
      isPositive: true
    };
  };
  
  const getRiskReduction = () => {
    return {
      value: 23.6,
      trend: "+5.2%",
      isPositive: true
    };
  };
  
  const getAccuracyImprovement = () => {
    return {
      value: 18.2,
      trend: "+3.4%",
      isPositive: true
    };
  };
  
  // Format currency function
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      maximumFractionDigits: 0
    }).format(value);
  };
  
  // Format percentage function
  const formatPercentage = (value: number) => {
    return `${value.toFixed(1)}%`;
  };
  
  return (
    <div className="flex min-h-screen flex-col">
      <DashboardHeader title="Business Insights" />
      <div className="container flex-1 items-start md:grid md:grid-cols-[220px_minmax(0,1fr)] md:gap-6 lg:grid-cols-[240px_minmax(0,1fr)] lg:gap-10">
        <DashboardSidebar />
        <main className="flex w-full flex-col overflow-hidden p-4 md:p-6">
          {/* Header Banner */}
          <Card className="mb-6 bg-gradient-to-r from-blue-50 to-indigo-50 border-blue-100">
            <CardContent className="p-6">
              <div className="flex flex-col md:flex-row md:items-center md:justify-between">
                <div>
                  <h2 className="text-xl font-bold text-blue-900 mb-2">
                    Business Impact Dashboard
                  </h2>
                  <p className="text-blue-800 max-w-2xl">
                    Track the ROI and business impact of ML models, with key metrics showing cost savings and premium optimization opportunities.
                  </p>
                </div>
                
                <div className="mt-4 md:mt-0 flex items-center gap-4">
                  <Select value={timeRange} onValueChange={setTimeRange}>
                    <SelectTrigger className="w-36 bg-white">
                      <SelectValue placeholder="Time Range" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="ytd">Year to Date</SelectItem>
                      <SelectItem value="q1">Q1 2023</SelectItem>
                      <SelectItem value="q2">Q2 2023</SelectItem>
                      <SelectItem value="q3">Q3 2023</SelectItem>
                      <SelectItem value="q4">Q4 2023</SelectItem>
                      <SelectItem value="custom">Custom Range</SelectItem>
                    </SelectContent>
                  </Select>
                  
                  <Button variant="outline" size="sm" className="bg-white">
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Refresh
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
          
          {/* Key Business Metrics Overview */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <Card>
              <CardContent className="p-4">
                <div className="flex justify-between items-center mb-2">
                  <div className="text-xs text-muted-foreground">ROI</div>
                  <Badge variant={getROI().isPositive ? "success" : "destructive"} className="text-xs">
                    {getROI().trend}
                  </Badge>
                </div>
                <div className="text-2xl font-bold flex items-center">
                  {getROI().value}%
                  {getROI().isPositive ? 
                    <ArrowUpRight className="h-4 w-4 text-green-500 ml-2" /> : 
                    <ArrowDownRight className="h-4 w-4 text-red-500 ml-2" />
                  }
                </div>
                <div className="text-xs text-muted-foreground mt-1">
                  Return on ML investment
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-4">
                <div className="flex justify-between items-center mb-2">
                  <div className="text-xs text-muted-foreground">Cost Savings</div>
                  <Badge variant={getCostSavings().isPositive ? "success" : "destructive"} className="text-xs">
                    {getCostSavings().trend}
                  </Badge>
                </div>
                <div className="text-2xl font-bold flex items-center">
                  {formatCurrency(getCostSavings().value)}
                  {getCostSavings().isPositive ? 
                    <ArrowUpRight className="h-4 w-4 text-green-500 ml-2" /> : 
                    <ArrowDownRight className="h-4 w-4 text-red-500 ml-2" />
                  }
                </div>
                <div className="text-xs text-muted-foreground mt-1">
                  Annual operational savings
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-4">
                <div className="flex justify-between items-center mb-2">
                  <div className="text-xs text-muted-foreground">Risk Reduction</div>
                  <Badge variant={getRiskReduction().isPositive ? "success" : "destructive"} className="text-xs">
                    {getRiskReduction().trend}
                  </Badge>
                </div>
                <div className="text-2xl font-bold flex items-center">
                  {formatPercentage(getRiskReduction().value)}
                  {getRiskReduction().isPositive ? 
                    <ArrowUpRight className="h-4 w-4 text-green-500 ml-2" /> : 
                    <ArrowDownRight className="h-4 w-4 text-red-500 ml-2" />
                  }
                </div>
                <div className="text-xs text-muted-foreground mt-1">
                  Risk exposure reduction
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-4">
                <div className="flex justify-between items-center mb-2">
                  <div className="text-xs text-muted-foreground">Accuracy Improvement</div>
                  <Badge variant={getAccuracyImprovement().isPositive ? "success" : "destructive"} className="text-xs">
                    {getAccuracyImprovement().trend}
                  </Badge>
                </div>
                <div className="text-2xl font-bold flex items-center">
                  {formatPercentage(getAccuracyImprovement().value)}
                  {getAccuracyImprovement().isPositive ? 
                    <ArrowUpRight className="h-4 w-4 text-green-500 ml-2" /> : 
                    <ArrowDownRight className="h-4 w-4 text-red-500 ml-2" />
                  }
                </div>
                <div className="text-xs text-muted-foreground mt-1">
                  Model accuracy gain
                </div>
              </CardContent>
            </Card>
          </div>
          
          {/* Business Impact Tabs */}
          <Card className="mb-6">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Business Impact Analysis</CardTitle>
                  <CardDescription>
                    Comprehensive analysis of ML model business impact
                  </CardDescription>
                </div>
                <Select value={modelType} onValueChange={setModelType}>
                  <SelectTrigger className="w-40">
                    <SelectValue placeholder="Model Type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="traditional">Traditional Model</SelectItem>
                    <SelectItem value="enhanced">Enhanced Model</SelectItem>
                    <SelectItem value="comparison">Comparison View</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="roi">
                <TabsList>
                  <TabsTrigger value="roi" className="flex items-center gap-1">
                    <Calculator className="h-4 w-4" />
                    ROI Analysis
                  </TabsTrigger>
                  <TabsTrigger value="cost-benefit" className="flex items-center gap-1">
                    <BarChart className="h-4 w-4" />
                    Cost-Benefit
                  </TabsTrigger>
                  <TabsTrigger value="premium" className="flex items-center gap-1">
                    <DollarSign className="h-4 w-4" />
                    Premium Optimization
                  </TabsTrigger>
                  <TabsTrigger value="risk" className="flex items-center gap-1">
                    <TrendingUp className="h-4 w-4" />
                    Risk Analysis
                  </TabsTrigger>
                </TabsList>
                
                {/* ROI Analysis Tab */}
                <TabsContent value="roi" className="pt-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-lg">ROI Breakdown</CardTitle>
                        <CardDescription>
                          Return on investment components
                        </CardDescription>
                      </CardHeader>
                      <CardContent className="pt-0">
                        <div className="h-60 flex items-center justify-center">
                          <p className="text-muted-foreground">ROI breakdown chart will be displayed here</p>
                        </div>
                      </CardContent>
                    </Card>
                    
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-lg">Investment Timeline</CardTitle>
                        <CardDescription>
                          ROI over implementation timeline
                        </CardDescription>
                      </CardHeader>
                      <CardContent className="pt-0">
                        <div className="h-60 flex items-center justify-center">
                          <p className="text-muted-foreground">ROI timeline chart will be displayed here</p>
                        </div>
                      </CardContent>
                    </Card>
                    
                    <Card className="md:col-span-2">
                      <CardHeader>
                        <CardTitle className="text-lg">ROI Components</CardTitle>
                        <CardDescription>
                          Cost and revenue impact breakdown
                        </CardDescription>
                      </CardHeader>
                      <CardContent className="pt-0">
                        <div className="space-y-4">
                          <div>
                            <div className="flex justify-between mb-1">
                              <span className="text-sm font-medium">Implementation Costs</span>
                              <span className="text-sm font-medium text-red-600">-$450,000</span>
                            </div>
                            <Progress value={45} className="h-2 bg-red-100 [&>*]:bg-red-500" />
                          </div>
                          <div>
                            <div className="flex justify-between mb-1">
                              <span className="text-sm font-medium">Maintenance Costs</span>
                              <span className="text-sm font-medium text-red-600">-$120,000/year</span>
                            </div>
                            <Progress value={12} className="h-2 bg-red-100 [&>*]:bg-red-500" />
                          </div>
                          <div>
                            <div className="flex justify-between mb-1">
                              <span className="text-sm font-medium">Loss Ratio Improvement</span>
                              <span className="text-sm font-medium text-green-600">+$850,000</span>
                            </div>
                            <Progress value={85} className="h-2 bg-green-100 [&>*]:bg-green-500" />
                          </div>
                          <div>
                            <div className="flex justify-between mb-1">
                              <span className="text-sm font-medium">Operational Efficiency</span>
                              <span className="text-sm font-medium text-green-600">+$390,000</span>
                            </div>
                            <Progress value={39} className="h-2 bg-green-100 [&>*]:bg-green-500" />
                          </div>
                          <div>
                            <div className="flex justify-between mb-1">
                              <span className="text-sm font-medium">Risk-Based Pricing</span>
                              <span className="text-sm font-medium text-green-600">+$670,000</span>
                            </div>
                            <Progress value={67} className="h-2 bg-green-100 [&>*]:bg-green-500" />
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </TabsContent>
                
                {/* Cost-Benefit Tab */}
                <TabsContent value="cost-benefit" className="pt-4">
                  <Card>
                    <CardHeader>
                      <CardTitle>Cost-Benefit Analysis

"use client"

import { DashboardHeader } from "@/components/dashboard-header"
import { DashboardSidebar } from "@/components/dashboard-sidebar"
import BusinessInsights from "@/components/business-insights"
import { DollarSign, PieChart, TrendingUp, Users } from "lucide-react"
import { SummaryCard } from "@/components/ui/summary-card"
import { ROIVisualization } from "@/components/roi-visualization"
import { CompetitiveAdvantageMetrics } from "@/components/competitive-advantage-metrics"
import { HistoricalPerformance } from "@/components/historical-performance"

export default function BusinessInsightsPage() {
  return (
    <div className="flex min-h-screen flex-col">
      <DashboardHeader title="Business Insights" />
      <div className="container flex-1 items-start md:grid md:grid-cols-[220px_minmax(0,1fr)] md:gap-6 lg:grid-cols-[240px_minmax(0,1fr)] lg:gap-10">
        <DashboardSidebar />
        <main className="flex w-full flex-col overflow-hidden p-4 md:p-6">
          {/* Key Metrics Summary */}
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4 mb-8">
            <SummaryCard
              title="Total Projected Savings"
              value="$2.4M"
              icon={DollarSign}
              trend="positive"
              changeValue="18.2%"
              tooltipText="Annual cost savings achieved through ML predictions compared to traditional methods."
              changeText="from last quarter"
              suffix=" annually"
            />
            <SummaryCard
              title="Loss Ratio Improvement"
              value="3.4"
              suffix="%"
              icon={TrendingUp}
              trend="positive"
              changeValue="0.8%"
              tooltipText="Reduction in the loss ratio through better predictive modeling and risk assessment."
            />
            <SummaryCard
              title="Pricing Accuracy"
              value="94.7"
              suffix="%"
              icon={PieChart}
              trend="positive"
              changeValue="2.3%"
              tooltipText="Accuracy of premium pricing models compared to actual losses."
            />
            <SummaryCard
              title="Customer Retention"
              value="89.2"
              suffix="%"
              icon={Users}
              trend="negative"
              changeValue="1.1%"
              tooltipText="Percentage of customers retained year-over-year after implementing ML-based pricing."
            />
          </div>
          
          {/* ROI Analysis Section */}
          <div className="mb-12">
            <div className="mb-6">
              <h2 className="text-2xl font-bold">Return on Investment Analysis</h2>
              <p className="text-muted-foreground">Detailed breakdown of financial benefits and cost savings from ML implementation</p>
            </div>
            <ROIVisualization />
          </div>

          {/* Competitive Advantages Section */}
          <div className="mb-12">
            <div className="mb-6">
              <h2 className="text-2xl font-bold">Competitive Advantages</h2>
              <p className="text-muted-foreground">How our ML solutions outperform traditional methods and industry standards</p>
            </div>
            <CompetitiveAdvantageMetrics />
          </div>

          {/* Historical Performance Section */}
          <div className="mb-12">
            <div className="mb-6">
              <h2 className="text-2xl font-bold">Model Performance Validation</h2>
              <p className="text-muted-foreground">Historical accuracy and stability metrics demonstrating consistent results</p>
            </div>
            <HistoricalPerformance />
          </div>

          {/* Detailed Business Insights */}
          <div className="mb-8">
            <div className="mb-6">
              <h2 className="text-2xl font-bold">Detailed Analysis</h2>
              <p className="text-muted-foreground">Comprehensive breakdown of predictions, trends, and risk assessments</p>
            </div>
            <BusinessInsights />
          </div>
        </main>
      </div>
    </div>
  )
} 
