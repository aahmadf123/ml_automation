"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Input } from "@/components/ui/input"
import { Download, Info, Share2, TrendingUp, DollarSign, BarChart, LineChart as LineChartIcon, Percentage } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart, BarChart as RechartBarChart, Bar, ReferenceLine, PieChart, Pie, Cell } from 'recharts'

// Sample data for premium projections
const premiumProjectionData = [
  { quarter: 'Q1 2023', actual: 8250000, projected: 8180000, lowerBound: 7950000, upperBound: 8410000 },
  { quarter: 'Q2 2023', actual: 8450000, projected: 8380000, lowerBound: 8150000, upperBound: 8610000 },
  { quarter: 'Q3 2023', actual: 8650000, projected: 8580000, lowerBound: 8350000, upperBound: 8810000 },
  { quarter: 'Q4 2023', actual: 8850000, projected: 8780000, lowerBound: 8550000, upperBound: 9010000 },
  // Projected for next year
  { quarter: 'Q1 2024', projected: 9050000, lowerBound: 8820000, upperBound: 9280000 },
  { quarter: 'Q2 2024', projected: 9280000, lowerBound: 9050000, upperBound: 9510000 },
  { quarter: 'Q3 2024', projected: 9520000, lowerBound: 9290000, upperBound: 9750000 },
  { quarter: 'Q4 2024', projected: 9780000, lowerBound: 9550000, upperBound: 10010000 },
];

// Growth projections by segment
const growthBySegmentData = [
  { name: 'Homeowners', currentValue: 5450000, projectedValue: 6250000, growth: 14.7 },
  { name: 'Auto', currentValue: 3250000, projectedValue: 3650000, growth: 12.3 },
  { name: 'Life', currentValue: 1850000, projectedValue: 2100000, growth: 13.5 },
  { name: 'Commercial', currentValue: 2700000, projectedValue: 3250000, growth: 20.4 },
  { name: 'Specialty', currentValue: 1150000, projectedValue: 1350000, growth: 17.4 },
];

// ROI metrics over time
const roiMetricsData = [
  { quarter: 'Q1 2023', actual: 19.8 },
  { quarter: 'Q2 2023', actual: 20.2 },
  { quarter: 'Q3 2023', actual: 21.5 },
  { quarter: 'Q4 2023', actual: 22.1 },
  { quarter: 'Q1 2024', projected: 22.5, optimistic: 23.2, conservative: 21.8 },
  { quarter: 'Q2 2024', projected: 23.0, optimistic: 23.9, conservative: 22.1 },
  { quarter: 'Q3 2024', projected: 23.5, optimistic: 24.5, conservative: 22.5 },
  { quarter: 'Q4 2024', projected: 24.2, optimistic: 25.3, conservative: 23.1 },
];

// Market distribution data
const marketDistributionData = [
  { name: 'Homeowners', value: 5450000, color: '#8884d8' },
  { name: 'Auto', value: 3250000, color: '#82ca9d' },
  { name: 'Life', value: 1850000, color: '#ffc658' },
  { name: 'Commercial', value: 2700000, color: '#ff8042' },
  { name: 'Specialty', value: 1150000, color: '#0088fe' },
];

// Business impact metrics
const businessImpactData = [
  { name: 'Revenue Growth', value: '+14.7%', trend: 'up' },
  { name: 'Cost Reduction', value: '-8.2%', trend: 'down' },
  { name: 'Customer Retention', value: '+5.3%', trend: 'up' },
  { name: 'Market Share', value: '+2.4%', trend: 'up' },
  { name: 'Profit Margin', value: '+3.8%', trend: 'up' },
];

export default function ProjectionsPage() {
  const [projectionModel, setProjectionModel] = useState('advanced')
  const [timeFrame, setTimeFrame] = useState('8')
  const [confidenceInterval, setConfidenceInterval] = useState('90')
  const [growthFactor, setGrowthFactor] = useState('12.5')
  
  // Format currency
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };
  
  // Format percentage
  const formatPercentage = (value: number) => {
    return `${value.toFixed(1)}%`;
  };
  
  // Calculate total projected premium
  const totalProjectedPremium = premiumProjectionData
    .slice(4) // Only projection quarters
    .reduce((sum, item) => sum + item.projected, 0);
  
  // Calculate potential growth
  const annualGrowthRate = 14.7; // 14.7% annual growth
  
  return (
    <div className="container mx-auto p-4 space-y-6">
      {/* Header with actions */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Premium Projections</h1>
          <p className="text-muted-foreground">
            Forecasting premium growth and business impact with confidence intervals
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
            Refresh Projections
          </Button>
        </div>
      </div>
      
      {/* Business impact metrics */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        {businessImpactData.map((metric, index) => (
          <Card key={index} className="relative">
            <CardContent className="p-4 flex flex-col items-center">
              <p className="text-sm font-medium text-center">{metric.name}</p>
              <div className={`text-2xl font-bold mt-1 ${metric.trend === 'up' ? 'text-green-600' : 'text-red-600'}`}>
                {metric.value}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
      
      {/* Main premium projection chart */}
      <Card className="relative">
        <CardHeader className="pb-2">
          <div className="flex justify-between items-center">
            <div>
              <CardTitle className="text-lg">Premium Projections</CardTitle>
              <CardDescription>
                Historical and projected premium with {confidenceInterval}% confidence interval
              </CardDescription>
            </div>
            <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
              Annual Growth: {annualGrowthRate}%
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={premiumProjectionData} margin={{ top: 10, right: 30, left: 20, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorPremiumUpper" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#82ca9d" stopOpacity={0.15}/>
                    <stop offset="95%" stopColor="#82ca9d" stopOpacity={0.05}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="quarter" />
                <YAxis tickFormatter={(value) => formatCurrency(value)} />
                <Tooltip formatter={(value) => formatCurrency(Number(value))} />
                <Legend />
                <Area type="monotone" dataKey="upperBound" stroke="none" fillOpacity={1} fill="url(#colorPremiumUpper)" name="Confidence Interval" />
                <Area type="monotone" dataKey="lowerBound" stroke="none" fillOpacity={0} fill="url(#colorPremiumUpper)" name="Confidence Interval" />
                <Line type="monotone" dataKey="actual" stroke="#4f46e5" strokeWidth={2} dot={{ r: 4 }} name="Actual Premium" activeDot={{ r: 6 }} />
                <Line type="monotone" dataKey="projected" stroke="#10b981" strokeWidth={2} dot={{ r: 4 }} name="Projected Premium" activeDot={{ r: 6 }} />
                <ReferenceLine x="Q4 2023" stroke="#666" strokeDasharray="3 3" label={{ value: 'Projection Start', position: 'top' }} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
        <CardFooter className="border-t p-4 flex justify-between">
          <div className="flex space-x-4">
            <div>
              <Select value={projectionModel} onValueChange={setProjectionModel}>
                <SelectTrigger className="w-36">
                  <SelectValue placeholder="Model" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="basic">Basic Linear</SelectItem>
                  <SelectItem value="advanced">Advanced ML</SelectItem>
                  <SelectItem value="conservative">Conservative</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <Select value={timeFrame} onValueChange={setTimeFrame}>
                <SelectTrigger className="w-36">
                  <SelectValue placeholder="Time Frame" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="4">4 Quarters</SelectItem>
                  <SelectItem value="8">8 Quarters</SelectItem>
                  <SelectItem value="12">12 Quarters</SelectItem>
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
                  <SelectItem value="90">90% Confidence</SelectItem>
                  <SelectItem value="95">95% Confidence</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          <div>
            <Button variant="outline" size="sm">
              <LineChartIcon className="mr-2 h-4 w-4" />
              Sensitivity Analysis
            </Button>
          </div>
        </CardFooter>
      </Card>
      
      {/* ROI and segment analysis */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Growth by Segment Analysis */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Growth by Segment</CardTitle>
            <CardDescription>
              Projected growth across different business segments
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <RechartBarChart data={growthBySegmentData} margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis tickFormatter={(value) => formatCurrency(value)} />
                  <Tooltip formatter={(value) => formatCurrency(Number(value))} />
                  <Legend />
                  <Bar dataKey="currentValue" fill="#4f46e5" name="Current Value" />
                  <Bar dataKey="projectedValue" fill="#10b981" name="Projected Value" />
                </RechartBarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
        
        {/* ROI Projection */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">ROI Projection</CardTitle>
            <CardDescription>
              Return on Investment projections by quarter
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={roiMetricsData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="quarter" />
                  <YAxis tickFormatter={(value) => `${value}%`} />
                  <Tooltip formatter={(value) => `${Number(value).toFixed(1)}%`} />
                  <Legend />
                  <Line type="monotone" dataKey="actual" stroke="#4f46e5" strokeWidth={2} dot={{ r: 4 }} name="Actual ROI" activeDot={{ r: 6 }} />
                  <Line type="monotone" dataKey="projected" stroke="#10b981" strokeWidth={2} dot={{ r: 4 }} name="Projected ROI" activeDot={{ r: 6 }} />
                  <Line type="monotone" dataKey="optimistic" stroke="#10b981" strokeDasharray="5 5" strokeWidth={1} dot={{ r: 3 }} name="Optimistic ROI" />
                  <Line type="monotone" dataKey="conservative" stroke="#10b981" strokeDasharray="5 5" strokeWidth={1} dot={{ r: 3 }} name="Conservative ROI" />
                  <ReferenceLine x="Q4 2023" stroke="#666" strokeDasharray="3 3" label={{ value: 'Projection Start', position: 'top' }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>
      
      {/* Additional insights */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Market Distribution */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Market Distribution</CardTitle>
            <CardDescription>
              Premium distribution across market segments
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-64 flex items-center justify-center">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={marketDistributionData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={90}
                    paddingAngle={2}
                    dataKey="value"
                    labelLine={false}
                    label={({ name, value, percent }) => `${name}: ${formatPercentage(percent * 100)}`}
                  >
                    {marketDistributionData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => formatCurrency(Number(value))} />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
        
        {/* Growth Metrics */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Premium Growth Metrics</CardTitle>
            <CardDescription>
              Key performance indicators for premium growth
            </CardDescription>
          </CardHeader>
          <CardContent className="p-6">
            <div className="grid grid-cols-2 gap-4">
              <div className="flex flex-col items-center justify-center">
                <p className="text-xs text-muted-foreground">Total Projected Premium</p>
                <p className="text-2xl font-bold">{formatCurrency(totalProjectedPremium)}</p>
              </div>
              <div className="flex flex-col items-center justify-center">
                <p className="text-xs text-muted-foreground">Annual Growth</p>
                <p className="text-2xl font-bold text-green-600">{annualGrowthRate}%</p>
              </div>
              <div className="flex flex-col items-center justify-center">
                <p className="text-xs text-muted-foreground">Risk-Adjusted Growth</p>
                <p className="text-2xl font-bold text-amber-600">{(annualGrowthRate * 0.85).toFixed(1)}%</p>
              </div>
              <div className="flex flex-col items-center justify-center">
                <p className="text-xs text-muted-foreground">Confidence Score</p>
                <p className="text-2xl font-bold">{parseInt(confidenceInterval)}%</p>
              </div>
            </div>
          </CardContent>
        </Card>
        
        {/* Growth Adjustment Controls */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Growth Adjustments</CardTitle>
            <CardDescription>
              Fine-tune growth projections for scenario planning
            </CardDescription>
          </CardHeader>
          <CardContent className="p-6 space-y-4">
            <div>
              <div className="flex justify-between mb-2">
                <span className="text-sm">Growth Factor</span>
                <span className="text-sm font-bold">{growthFactor}%</span>
              </div>
              <Slider 
                value={[parseFloat(growthFactor)]} 
                min={5} 
                max={20} 
                step={0.5}
                onValueChange={(values) => setGrowthFactor(values[0].toString())} 
              />
            </div>
            <div>
              <div className="flex justify-between mb-2">
                <span className="text-sm">Risk Adjustment</span>
                <span className="text-sm font-bold">{(parseFloat(growthFactor) * 0.85).toFixed(1)}%</span>
              </div>
              <Slider 
                value={[85]} 
                min={50} 
                max={100} 
                step={5}
                disabled
              />
            </div>
            <div className="pt-2">
              <Button variant="default" size="sm" className="w-full">
                <TrendingUp className="mr-2 h-4 w-4" />
                Recalculate Projections
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

