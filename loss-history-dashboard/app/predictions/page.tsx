"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Input } from "@/components/ui/input"
import { Download, Info, Share2, TrendingUp, DollarSign, BarChart } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart, BarChart as RechartBarChart, Bar, ReferenceLine } from 'recharts'

// Sample data for loss prediction
const lossHistoryData = [
  { month: 'Jan 2023', actual: 3250000, predicted: 3180000, lowerBound: 2950000, upperBound: 3410000 },
  { month: 'Feb 2023', actual: 3105000, predicted: 3220000, lowerBound: 2980000, upperBound: 3460000 },
  { month: 'Mar 2023', actual: 3320000, predicted: 3260000, lowerBound: 3020000, upperBound: 3500000 },
  { month: 'Apr 2023', actual: 3480000, predicted: 3380000, lowerBound: 3130000, upperBound: 3630000 },
  { month: 'May 2023', actual: 3520000, predicted: 3510000, lowerBound: 3250000, upperBound: 3770000 },
  { month: 'Jun 2023', actual: 3610000, predicted: 3650000, lowerBound: 3380000, upperBound: 3920000 },
  { month: 'Jul 2023', actual: 3780000, predicted: 3750000, lowerBound: 3470000, upperBound: 4030000 },
  { month: 'Aug 2023', actual: 3850000, predicted: 3820000, lowerBound: 3535000, upperBound: 4105000 },
  { month: 'Sep 2023', actual: 3920000, predicted: 3880000, lowerBound: 3590000, upperBound: 4170000 },
  { month: 'Oct 2023', actual: 3980000, predicted: 3940000, lowerBound: 3645000, upperBound: 4235000 },
  { month: 'Nov 2023', actual: 4050000, predicted: 4010000, lowerBound: 3710000, upperBound: 4310000 },
  { month: 'Dec 2023', actual: 4130000, predicted: 4080000, lowerBound: 3775000, upperBound: 4385000 },
  // Forecast for next 6 months
  { month: 'Jan 2024', predicted: 4150000, lowerBound: 3840000, upperBound: 4460000 },
  { month: 'Feb 2024', predicted: 4210000, lowerBound: 3895000, upperBound: 4525000 },
  { month: 'Mar 2024', predicted: 4280000, lowerBound: 3960000, upperBound: 4600000 },
  { month: 'Apr 2024', predicted: 4350000, lowerBound: 4025000, upperBound: 4675000 },
  { month: 'May 2024', predicted: 4420000, lowerBound: 4090000, upperBound: 4750000 },
  { month: 'Jun 2024', predicted: 4490000, lowerBound: 4155000, upperBound: 4825000 },
];

// Risk categories data
const riskCategoryData = [
  { category: 'Fire', predicted: 1850000, change: 8.5, impact: 'high' },
  { category: 'Water', predicted: 1320000, change: 12.3, impact: 'high' },
  { category: 'Theft', predicted: 620000, change: -3.2, impact: 'medium' },
  { category: 'Liability', predicted: 510000, change: 5.8, impact: 'medium' },
  { category: 'Wind', predicted: 450000, change: 15.7, impact: 'high' },
  { category: 'Other', predicted: 250000, change: 2.1, impact: 'low' },
];

// ROI impact metrics
const roiMetrics = [
  { name: 'Adjusted Premiums', value: '+$1.2M' },
  { name: 'Reserved Capital', value: '-$820K' },
  { name: 'Risk Mitigation', value: '+$650K' },
  { name: 'Operational Savings', value: '+$430K' },
  { name: 'Total ROI', value: '+$1.46M' },
];

export default function PredictionsPage() {
  const [forecastPeriod, setForecastPeriod] = useState('6')
  const [confidenceInterval, setConfidenceInterval] = useState('95')
  const [modelSelection, setModelSelection] = useState('model4')
  const [seasonalAdjustment, setSeasonalAdjustment] = useState('true')
  
  // Format currency
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };
  
  // Calculate total predicted loss
  const totalPredictedLoss = lossHistoryData
    .slice(12) // Only forecast months
    .reduce((sum, item) => sum + item.predicted, 0);
  
  // Calculate potential savings based on model improvements
  const potentialSavings = totalPredictedLoss * 0.08; // 8% savings through improved prediction
  
  return (
    <div className="container mx-auto p-4 space-y-6">
      {/* Header section */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Loss Predictions</h1>
          <p className="text-muted-foreground">
            Forecasting potential losses with advanced ML models and confidence intervals
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" className="hidden md:flex">
            <Share2 className="mr-2 h-4 w-4" />
            Share
          </Button>
          <Button variant="outline" size="sm" className="hidden md:flex">
            <Download className="mr-2 h-4 w-4" />
            Export
          </Button>
          <Button size="sm">
            <TrendingUp className="mr-2 h-4 w-4" />
            Run Prediction
          </Button>
        </div>
      </div>
      
      {/* Main forecast chart */}
      <Card className="relative">
        <CardHeader className="pb-2">
          <div className="flex justify-between items-center">
            <div>
              <CardTitle className="text-lg">Loss Forecasting</CardTitle>
              <CardDescription>
                Historical and predicted losses with {confidenceInterval}% confidence interval
              </CardDescription>
            </div>
            <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200">
              Model: {modelSelection === 'model4' ? 'Fast Decay (17.9% more accurate)' : 'Traditional'}
                    <Card>
          <CardHeader>
            <CardTitle className="text-lg">Model Comparison</CardTitle>
            <CardDescription>
              Enhanced Model4 vs. Traditional Model1 performance
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              <div className="flex justify-between items-center">
                <div>
                  <h3 className="text-sm font-medium">Accuracy Improvement</h3>
                  <p className="text-3xl font-bold text-green-600">+17.9%</p>
                </div>
                <div>
                  <h3 className="text-sm font-medium">Prediction Error</h3>
                  <p className="text-3xl font-bold text-green-600">-22.4%</p>
                </div>
                <div>
                  <h3 className="text-sm font-medium">Confidence Score</h3>
                  <p className="text-3xl font-bold">95%</p>
                </div>
              </div>
              
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span>Model4 (Fast Decay)</span>
                    <span>94.8% accuracy</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2.5">
                    <div className="bg-green-600 h-2.5 rounded-full" style={{ width
              
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span>Model4 (Fast Decay)</span>
                    <span>94.8% accuracy</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2.5">
                    <div className="bg-green-600 h-2.5 rounded-full" style={{ width: '94.8%' }}></div>
                  </div>
                </div>
                
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span>Model1 (Traditional)</span>
                    <span>76.9% accuracy</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2.5">
                    <div className="bg-blue-600 h-2.5 rounded-full" style={{ width: '76.9%' }}></div>
                  </div>
                </div>
              </div>
              
              <div className="bg-blue-50 p-4 rounded-lg border border-blue-100">
                <h3 className="text-sm font-medium text-blue-800 mb-2">Key Performance Benefits</h3>
                <ul className="text-xs text-blue-700 space-y-1">
                  <li>• Fast Decay model responds more quickly to recent loss patterns</li>
                  <li>• 17.9% higher accuracy reduces capital requirements by up to 8%</li>
                  <li>• Improved risk pricing leads to $1.2M premium optimization</li>
                  <li>• Reduces catastrophic loss surprises by 32%</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
      
      {/* Business Impact Summary */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Business Impact Summary</CardTitle>
          <CardDescription>
            Financial impact of improved loss predictions over the next {forecastPeriod} months
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="space-y-4">
              <div className="flex justify-between">
                <span className="text-sm">Total Predicted Loss:</span>
                <span className="text-sm font-medium">{formatCurrency(totalPredictedLoss)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm">Potential Savings:</span>
                <span className="text-sm font-medium text-green-600">{formatCurrency(potentialSavings)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm">Model Selection:</span>
                <span className="text-sm font-medium">{modelSelection === 'model4' ? 'Fast Decay (Enhanced)' : 'Traditional'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm">Forecast Range:</span>
                <span className="text-sm font-medium">{forecastPeriod} months</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm">Confidence Level:</span>
                <span className="text-sm font-medium">{confidenceInterval}%</span>
              </div>
            </div>
            
            <div className="col-span-2 bg-gray-50 p-4 rounded-lg border border-gray-200 dark:bg-gray-800 dark:border-gray-700">
              <h3 className="font-medium mb-2">Recommended Actions</h3>
              <ul className="text-sm space-y-2">
                <li className="flex items-start">
                  <div className="bg-green-100 text-green-700 p-1 rounded-full mt-0.5 mr-2">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                  </div>
                  <div>
                    <strong>Adjust reserve capital</strong>: Optimize capital allocation based on the {formatCurrency(potentialSavings)} potential savings identified
                  </div>
                </li>
                <li className="flex items-start">
                  <div className="bg-green-100 text-green-700 p-1 rounded-full mt-0.5 mr-2">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                  </div>
                  <div>
                    <strong>Focus on high-impact risk categories</strong>: Prioritize mitigation for Fire (+8.5%) and Water (+12.3%) damage losses
                  </div>
                </li>
                <li className="flex items-start">
                  <div className="bg-green-100 text-green-700 p-1 rounded-full mt-0.5 mr-2">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                  </div>
                  <div>
                    <strong>Adjust premium structure</strong>: Use Model4's enhanced accuracy to refine risk-based pricing
                  </div>
                </li>
              </ul>
            </div>
          </div>
        </CardContent>
        <CardFooter className="border-t p-4 flex justify-end">
          <Button variant="outline" size="sm" className="mr-2">
            <BarChart className="mr-2 h-4 w-4" />
            View Full Report
          </Button>
          <Button
odel" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="model1">Traditional</SelectItem>
                  <SelectItem value="model4">Fast Decay</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <Select value={forecastPeriod} onValueChange={setForecastPeriod}>
                <SelectTrigger className="w-36">
                  <SelectValue placeholder="Forecast Period" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="3">3 Months</SelectItem>
                  <SelectItem value="6">6 Months</SelectItem>
                  <SelectItem value="12">12 Months</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <Select value={confidenceInterval} onValueChange={setConfidenceInterval}>
                <SelectTrigger className="w-36">
                  <SelectValue placeholder="Confidence" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="80">80% Confidence</SelectItem>
                  <SelectItem value="95">95% Confidence</SelectItem>
                  <SelectItem value="99">99% Confidence</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <Select value={seasonalAdjustment} onValueChange={setSeasonalAdjustment}>
                <SelectTrigger className="w-36">
                  <SelectValue placeholder="Seasonal Adjustment" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="true">Seasonal Adjusted</SelectItem>
                  <SelectItem value="false">Raw Data</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={lossHistoryData} margin={{ top: 10, right: 30, left: 20, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorUpper" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#8884d8" stopOpacity={0.15}/>
                    <stop offset="95%" stopColor="#8884d8" stopOpacity={0.05}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis tickFormatter={(value) => formatCurrency(value)} />
                <Tooltip formatter={(value) => formatCurrency(Number(value))} />
                <Legend />
                <Area type="monotone" dataKey="upperBound" stroke="none" fillOpacity={1} fill="url(#colorUpper)" name="Confidence Interval" />
                <Area type="monotone" dataKey="lowerBound" stroke="none" fillOpacity={0} fill="url(#colorUpper)" name="Confidence Interval" />
                <Line type="monotone" dataKey="actual" stroke="#4f46e5" strokeWidth={2} dot={{ r: 4 }} name="Actual Losses" activeDot={{ r: 6 }} />
                <Line type="monotone" dataKey="predicted" stroke="#fb923c" strokeWidth={2} dot={{ r: 4 }} name="Predicted Losses" activeDot={{ r: 6 }} />
                <ReferenceLine x="Dec 2023" stroke="#666" strokeDasharray="3 3" label={{ value: 'Forecast Start', position: 'top' }} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
        <CardFooter className="border-t p-4 flex justify-between">
          <div className="flex space-x-4">
            <div>
              <Select value={modelSelection} onValueChange={setModelSelection}>
                <SelectTrigger className="w-36">
                  <SelectValue placeholder="Model" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="model1">Traditional</SelectItem>
                  <SelectItem value="model4">Fast Decay</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <Select value={forecastPeriod} onValueChange={setForecastPeriod}>
                <SelectTrigger className="w-36">
                  <SelectValue placeholder="Forecast Period" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="3">3 Months</SelectItem>
                  <SelectItem value="6">6 Months</SelectItem>
                  <SelectItem value="12">12 Months</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <Select value={confidenceInterval} onValueChange={setConfidenceInterval}>
                <SelectTrigger className="w-36">
                  <SelectValue placeholder="Confidence" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="80">80% Confidence</SelectItem>
                  <SelectItem value="95">95% Confidence</SelectItem>
                  <SelectItem value="99">99% Confidence</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

