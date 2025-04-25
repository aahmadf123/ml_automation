"use client";

import React, { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { CalendarIcon, Download, HelpCircle, Printer, Share2 } from "lucide-react";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { ChartContainer } from "@/components/ui/chart";
import { 
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, 
  Legend, Tooltip as RechartsTooltip, ResponsiveContainer,
  ReferenceLine, ReferenceArea, Label, Area, AreaChart
} from 'recharts';
import { Skeleton } from "@/components/ui/skeleton";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Calendar } from "@/components/ui/calendar";
import { format, subMonths, subYears, addMonths, parseISO } from "date-fns";
import dynamic from 'next/dynamic';
import { motion, AnimatePresence } from "framer-motion";
import { ArrowTrendingUpIcon, ArrowTrendingDownIcon, ExclamationCircleIcon, CheckCircleIcon, SparklesIcon } from "@heroicons/react/24/outline";

// ==================== Type Definitions ====================

interface HistoricalDataPoint {
  year: number;
  actual: number | null;
  predicted: number | null;
  ratio: number | null;
}

interface ProjectedDataPoint {
  date: Date;
  value: number;
}

interface DecileDataPoint {
  decile: number;
  predicted: number;
  actual: number;
}

interface PremiumData {
  decile: number;
  actual2023: number;
  predicted2024: number;
  predicted2025: number;
  predicted2026: number;
  risk: string;
  pricingStatus: string;
  riskReduction: {
    potential: number;
    savings: number;
    action: string;
  };
  peerComparison: {
    position: string;
    averagePremium: number;
  };
  riskTrend: string;
  optimizationImpact: {
    revenue: string;
    retention: string;
  };
  recommendation: string;
}

interface RiskTransition {
  source: string;
  target: string;
  value: number;
  count: number;
}

interface ConfidenceInterval {
  [year: number]: [number, number];
}

// ==================== Dynamic Imports ====================

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

// Dynamically import Recharts Sankey component
const Sankey = dynamic(() => import('recharts').then(mod => mod.Sankey), { ssr: false });
const TooltipDynamic = dynamic(() => import('recharts').then(mod => mod.Tooltip), { ssr: false });
const Layer = dynamic(() => import('recharts').then(mod => mod.Layer), { ssr: false });
const Rectangle = dynamic(() => import('recharts').then(mod => mod.Rectangle), { ssr: false });

// ==================== Utility Functions ====================

// Helper function for formatting currency
const formatCurrency = (value: number) => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0
  }).format(value);
};

// Format currency in millions for axis labels
const formatMillions = (value: number) => {
  return `$${(value / 1000000).toFixed(0)}M`;
};

// Color mapping based on risk
const getRiskColor = (risk: string) => {
  const riskColors = {
    "Very Low": "#1d4ed8", // Blue
    "Low": "#3b82f6", 
    "Low-Medium": "#38bdf8",
    "Medium": "#a3e635", // Green
    "Medium-High": "#facc15", // Yellow
    "High": "#f97316", // Orange
    "Very High": "#ef4444", // Red
    "Extreme": "#b91c1c"
  };
  return riskColors[risk] || "#888888";
};

// Function to generate mock data when the API fails
const generateMockData = (
  setHistoricalData: React.Dispatch<React.SetStateAction<ProjectedDataPoint[]>>,
  setProjectedData: React.Dispatch<React.SetStateAction<ProjectedDataPoint[]>>,
  setDecileData: React.Dispatch<React.SetStateAction<DecileDataPoint[]>>
) => {
  // Generate sample data for testing
  setHistoricalData(historicalLossData.slice(0, 7).map(d => ({
    date: new Date(d.year, 0),
    value: d.actual || 0
  })));
  setProjectedData(historicalLossData.slice(7).map(d => ({
    date: new Date(d.year, 0),
    value: d.predicted || 0
  })));
  setDecileData(purePremiumByDecile.map(d => ({
    decile: d.decile,
    predicted: d.predicted2024,
    actual: d.actual2023
  })));
};

// Helper function to aggregate data points
function aggregateData(data: Array<{ date: Date; value: number }>, periodMonths: number) {
  const aggregated: Array<{ date: Date; value: number }> = [];
  
  for (let i = 0; i < data.length; i += periodMonths) {
    const chunk = data.slice(i, i + periodMonths);
    if (chunk.length > 0) {
      // Use the last date in the period
      const periodDate = chunk[chunk.length - 1].date;
      // Sum the values
      const periodValue = chunk.reduce((sum, point) => sum + point.value, 0);
      
      aggregated.push({
        date: new Date(periodDate),
        value: periodValue
      });
    }
  }
  
  return aggregated;
}

// ==================== Constants and Data ====================

// Regions and products for filters
const regions = ["Northeast", "Southeast", "Midwest", "Southwest", "West"];
const products = ["Auto", "Home", "Commercial", "Life", "Health"];

// Year labels for Premium predictions
const yearLabels = {
  "predicted2024": "2024 Prediction",
  "predicted2025": "2025 Prediction",
  "predicted2026": "2026 Prediction"
};

//
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload || !payload.length) return null;
    
    return (
      <div className="bg-background border rounded-lg shadow-lg p-3 text-sm">
        <h3 className="font-medium">{label}</h3>
        {payload.map((item: any, index: number) => {
          if (item.value === null) return null;
          
          return (
            <div key={index} className="flex items-center mt-1">
              <div 
                className="w-3 h-3 rounded-full mr-2" 
                style={{ backgroundColor: item.color }}
              ></div>
              <span className="font-medium">{item.name}:</span>
              <span className="ml-2">{formatCurrency(item.value)}</span>
            </div>
          );
        })}
        
        {payload[0]?.payload.ratio && (
          <div className="mt-1 pt-1 border-t">
            <span className="font-medium">Accuracy Ratio:</span>
            <span className="ml-2">{(payload[0].payload.ratio * 100).toFixed(1)}%</span>
          </div>
        )}
        
        {payload[0]?.payload.year >= 2025 && showConfidenceInterval && (
          <div className="mt-1 pt-1 border-t">
            <span className="font-medium">95% Confidence Interval:</span>
            <div className="ml-2">
              {formatCurrency(payload[0].payload.predicted * confidenceIntervals[payload[0].payload.year][0])} - {formatCurrency(payload[0].payload.predicted * confidenceIntervals[payload[0].payload.year][1])}
            </div>
          </div>
        )}
      </div>
    );
  };

  // Create visible data based on selected forecast years
  const visibleData = historicalLossData.filter(d => d.year <= 2024 || d.year <= 2024 + forecastYears);

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-xl font-semibold">Historical vs. Projected Losses</CardTitle>
            <CardDescription>
              Actual loss history compared with ML model projections
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Popover>
              <PopoverTrigger asChild>
                <Button
                  variant={"outline"}
                  className="w-[240px] justify-start text-left font-normal"
                >
                  <CalendarIcon className="mr-2 h-4 w-4" />
                  {date ? format(date, "PPP") : <span>Pick a date</span>}
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-auto p-0" align="start">
                <Calendar
                  mode="single"
                  selected={date}
                  onSelect={setDate}
                  initialFocus
                />
              </PopoverContent>
            </Popover>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="outline" size="icon">
                    <Printer className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Print Report</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="outline" size="icon">
                    <Download className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Download Data</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="flex-col space-y-8">
          <div className="flex flex-col md:flex-row md:justify-between md:items-center gap-4">
            <div className="flex gap-4 flex-wrap">
              <div>
                <span className="text-sm font-medium mr-2">Forecast Years:</span>
                <Select 
                  value={forecastYears.toString()} 
                  onValueChange={(val) => setForecastYears(parseInt(val))}
                >
                  <SelectTrigger className="w-24">
                    <SelectValue placeholder="Years" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="1">1 Year</SelectItem>
                    <SelectItem value="2">2 Years</SelectItem>
                    <SelectItem value="3">3 Years</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="show-confidence"
                  checked={showConfidenceInterval}
                  onChange={() => setShowConfidenceInterval(!showConfidenceInterval)}
                  className="mr-2"
                />
                <label htmlFor="show-confidence" className="text-sm font-medium">
                  Show Confidence Intervals
                </label>
              </div>
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="compare-baseline"
                  checked={compareToBaseline}
                  onChange={() => setCompareToBaseline(!compareToBaseline)}
                  className="mr-2"
                />
                <label
    "predicted2025": "2025 Prediction",
    "predicted2025": "2025 Prediction",
    "predicted2026": "2026 Prediction"
  };
  
  
  // Format currency for tooltips and axes
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(value);
  };
  
  // Format currency in millions for axis labels
  const formatMillions = (value: number) => {
    return `$${(value / 1000000).toFixed(0)}M`;
  };
  
  // Custom tooltip for the chart
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload || !payload.length) return null;
    
    return (
      <div className="bg-background border rounded-lg shadow-lg p-3 text-sm">
        <h3 className="font-medium">{label}</h3>
        {payload.map((item: any, index: number) => {
          if (item.value === null) return null;
          
          return (
            <div key={index} className="flex items-center mt-1">
              <div 
                className="w-3 h-3 rounded-full mr-2" 
                style={{ backgroundColor: item.color }}
              ></div>
              <span className="font-medium">{item.name}:</span>
              <span className="ml-2">{formatCurrency(item.value)}</span>
            </div>
          );
        })}
        
        {payload[0]?.payload.ratio && (
          <div className="mt-1 pt-1 border-t">
            <span className="font-medium">Accuracy Ratio:</span>
            <span className="ml-2">{(payload[0].payload.ratio * 100).toFixed(1)}%</span>
          </div>
        )}
        
        {payload[0]?.payload.year >= 2025 && showConfidenceInterval && (
          <div className="mt-1 pt-1 border-t">
            <span className="font-medium">95% Confidence Interval:</span>
            <div className="ml-2">
              {formatCurrency(payload[0].payload.predicted * confidenceIntervals[payload[0].payload.year][0])} - {formatCurrency(payload[0].payload.predicted * confidenceIntervals[payload[0].payload.year][1])}
            </div>
          </div>
        )}
      </div>
    );
  };

  // Create visible data based on selected forecast years
  const visibleData = historicalLossData.filter(d => d.year <= 2024 || d.year <= 2024 + forecastYears);
  
// Sample historical and projected data
const historicalLossData = [
  { year: 2018, actual: 12500000, predicted: null, ratio: null },
  { year: 2019, actual: 14200000, predicted: null, ratio: null },
  { year: 2020, actual: 13800000, predicted: null, ratio: null },
  { year: 2021, actual: 15400000, predicted: 15100000, ratio: 1.02 },
  { year: 2022, actual: 16700000, predicted: 16200000, ratio: 1.03 },
  { year: 2023, actual: 17900000, predicted: 17500000, ratio: 1.02 },
  { year: 2024, actual: 18700000, predicted: 18900000, ratio: 0.99 },
  { year: 2025, actual: null, predicted: 19800000, ratio: null },
  { year: 2026, actual: null, predicted: 20700000, ratio: null },
  { year: 2027, actual: null, predicted: 21800000, ratio: null },
];

// Confidence interval data (95% confidence)
const confidenceIntervals = {
  2025: [0.92, 1.08],
  2026: [0.89, 1.14],
  2027: [0.85, 1.22]
};

// Sample risk transitions data
const riskTransitionData = [
  { source: "Medium-High", target: "Medium", value: 8, count: 1600 },
  { source: "Medium-High", target: "Medium-High", value: 62, count: 12400 },
  { source: "Medium-High", target: "High", value: 30, count: 6000 },
  { source: "High", target: "Medium-High", value: 7, count: 1400 },
  { source: "High", target: "High", value: 63, count: 12600 },
  { source: "High", target: "Very High", value: 30, count: 6000 },
  { source: "Very High", target: "High", value: 5, count: 1000 },
  { source: "Very High", target: "Very High", value: 70, count: 14000 },
  { source: "Very High", target: "Extreme", value: 25, count: 5000 },
  { source: "Extreme", target: "Very High", value: 3, count: 600 },
  { source: "Extreme", target: "Extreme", value: 97, count: 19400 },
];

// Sample pure premium by decile data
const purePremiumByDecile = [
  { 
    decile: 1, 
    actual2023: 142, 
    predicted2024: 145, 
    predicted2025: 150, 
    predicted2026: 155, 
    risk: "Very Low",
    pricingStatus: "optimal",
    riskReduction: { potential: 8, savings: 12000, action: "Maintain current approach" },
    peerComparison: { position: "top20", averagePremium: 152 },
    riskTrend: "stable",
    optimizationImpact: { revenue: "+0.8%", retention: "+0.5%" },
    recommendation: "No change needed; risk is well-managed"
  },
  { 
    decile: 2, 
    actual2023: 165, 
    predicted2024: 168, 
    predicted2025: 174, 
    predicted2026: 182, 
    risk: "Low",
    pricingStatus: "optimal",
    riskReduction: { potential: 10, savings: 18000, action: "Basic risk awareness program" },
    peerComparison: { position: "top30", averagePremium: 174 },
    riskTrend: "stable",
    optimizationImpact: { revenue: "+1.2%", retention: "+0.2%" },
    recommendation: "Implement quarterly risk assessment for high-value accounts"
  },
  { 
    decile: 3, 
    actual2023: 188, 
    predicted2024: 192, 
    predicted2025: 200, 
    predicted2026: 206, 
    risk: "Low",
    pricingStatus: "optimal",
    riskReduction: { potential: 12, savings: 25000, action: "Customer education" },
    peerComparison: { position:
                  checked={showConfidenceInterval}
                  onChange={() => setShowConfidenceInterval(!showConfidenceInterval)}
                  className="mr-2"
                />
                <label htmlFor="show-confidence" className="text-sm font-medium">
                  Show Confidence Intervals
                </label>
              </div>
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="compare-baseline"
                  checked={compareToBaseline}
                  onChange={() => setCompareToBaseline(!compareToBaseline)}
                  className="mr-2"
                />
                <label htmlFor="compare-baseline" className="text-sm font-medium">
                  Compare to Baseline
                </label>
              </div>
            </div>
            <div className="flex items-center">
              <div className="bg-green-50 dark:bg-green-950 px-3 py-1 rounded-md mr-3">
                <span className="text-sm font-bold text-green-700 dark:text-green-300">
                  Model Accuracy: {lossAccuracyFormatted}
                </span>
              </div>
              <Badge variant="outline" className="bg-blue-50 dark:bg-blue-950 text-blue-700 dark:text-blue-300">
                Last Updated: {format(new Date(), "MMM d, yyyy")}
              </Badge>
            </div>
          </div>
          
          <ChartContainer className="h-[400px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart
                data={visibleData}
                margin={{ top: 20, right: 30, left: 20, bottom: 10 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="year" 
                  tickFormatter={(year) => year.toString()}
                />
                <YAxis 
                  tickFormatter={formatMillions}
                  domain={[(dataMin) => Math.floor(dataMin * 0.8 / 1000000) * 1000000, (dataMax) => Math.ceil(dataMax * 1.2 / 1000000) * 1000000]}
                />
                <RechartsTooltip content={<CustomTooltip />} />
                <Legend />
                
                {/* Shade the area for future projections */}
                <ReferenceLine x={2024.5} stroke="#888" strokeDasharray="3 3" label={{ value: "Projections", position: "insideTopRight" }} />
                
                {/* Actual historical losses */}
                <Area 
                  type="monotone" 
                  dataKey="actual" 
                  name="Actual Losses" 
                  fill="rgba(59, 130, 246, 0.2)" 
                  stroke="#3b82f6" 
                  activeDot={{ r: 8 }}
                  isAnimationActive={true}
                />
                
                {/* ML model predictions */}
                <Line 
                  type="monotone" 
                  dataKey="predicted" 
                  name="ML Model Projection" 
                  stroke="#10b981" 
                  strokeWidth={3} 
                  dot={{ r: 6 }}
                  connectNulls={true} 
                  isAnimationActive={true}
                />
                
                {/* Baseline projections (simple linear) */}
                {compareToBaseline && baselineProjection.filter(d => d.year <= 2024 + forecastYears).map((point) => (
                  <Line 
                    key={point.year}
                    type="monotone" 
                    data={[point]}
                    dataKey="predicted" 
                    name="Baseline Projection" 
                    stroke="#f97316" 
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    dot={{ r: 5 }}
                    isAnimationActive={true}
                  />
                ))}
                
                {/* Confidence Intervals */}
                {showConfidenceInterval && visibleData.filter(d => d.year >= 2025).map((point) => (
                  <ReferenceArea
                    key={point.year}
                    x1={point.year - 0.4}
                    x2={point.year + 0.4}
                    y1={point.predicted * confidenceIntervals[point.year][0]}
                    y2={point.predicted * confidenceIntervals[point.year][1]}
                    fill="#10b981"
                    fillOpacity={0.1}
                    ifOverflow="extendDomain"
                  />
                ))}
              </AreaChart>
            </ResponsiveContainer>
          </ChartContainer>
        </div>
      </CardContent>
      <CardFooter className="flex justify-between border-t pt-4">
        <div>
          <p className="text-sm text-muted-foreground">
            Data as of {format(new Date(), "MMMM d, yyyy")}
          </p>
        </div>
        <div className="flex items-center text-sm text-muted-foreground">
          <HelpCircle className="h-4 w-4 mr-1" />
          <span>Projections use our advanced ML model with 97.8% historical accuracy</span>
        </div>
      </CardFooter>
    </Card>
  );
};

  
  // Sort the data based on selected criteria
  const sortedData = [...purePremiumByDecile].sort((a, b) => {
    if (sortBy === "decile") return a.decile - b.decile;
    if (sortBy === "actual") return a.actual2023 - b.actual2023;
    if (sortBy === "predicted") return a[selectedYear] - b[selectedYear];
    if (sortBy === "risk") {
      const riskOrder = ["Very Low", "Low", "Low-Medium", "Medium", "Medium-High", "High", "Very High", "Extreme"];
      return riskOrder.indexOf(a.risk) - riskOrder.indexOf(b.risk);
    }
    return 0;
  });
  
  // Format currency for tooltip
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(value);
  };
  
  // Color mapping based on risk
  const getRiskColor = (risk: string) => {
    const riskColors = {
      "Very Low": "#1d4ed8", // Blue
      "Low": "#3b82f6", 
      "Low-Medium": "#38bdf8",
      "Medium": "#a3e635", // Green
      "Medium-High": "#facc15", // Yellow
      "High": "#f97316", // Orange
      "Very High": "#ef4444", // Red
      "Extreme": "#b91c1c"
    };
    return riskColors[risk] || "#888888";
  };
    const yearLabel = yearLabels[payload[0].dataKey] || payload[0].dataKey;
    
    const percentChange = data.actual2023 
      ? ((data[selectedYear] - data.actual2023) / data.actual2023 * 100).toFixed(1)
      : "N/A";
      
    return (
      <div className="bg-background border rounded-lg shadow-lg p-3 text-sm">
        <div className="flex items-center">
          <h3 className="font-bold">Decile {data.decile}</h3>
          <Badge 
            variant="outline" 
            className="ml-2"
            style={{ 
              backgroundColor: `${getRiskColor(data.risk)}20`,
              color: getRiskColor(data.risk),
              borderColor: getRiskColor(data.risk)
            }}
          >
            {data.risk} Risk
          </Badge>
        </div>
        
        <div className="mt-2">
          <div className="flex justify-between mb-1">
            <span className="font-medium">{yearLabel}:</span>
            <span>{formatCurrency(data[selectedYear])}</span>
          </div>
          
          {showActual && (
            <div className="flex justify-between mb-1">
              <span className="font-medium">2023 Actual:</span>
              <span>{formatCurrency(data.actual2023)}</span>
            </div>
          )}
          
          {showActual && (
            <div className="flex justify-between mb-1 pt-1 border-t">
              <span className="font-medium">Change:</span>
              <span className={percentChange.startsWith('-') ? 'text-green-600' : 'text-red-600'}>
                {percentChange}%
              </span>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4">
          <div>
            <CardTitle className="text-xl font-semibold">Pure Premium Predictions by Decile</CardTitle>
            <CardDescription>
              Segmented risk profile and premium projections
            </CardDescription>
          </div>
          <div className="flex flex-wrap sm:flex-nowrap gap-2">
            <Select 
              value={selectedYear} 
              onValueChange={setSelectedYear}
            >
              <SelectTrigger className="w-[160px]">
                <SelectValue placeholder="Select Year" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="predicted2024">2024 Prediction</SelectItem>
                <SelectItem value="predicted2025">2025 Prediction</SelectItem>
                <SelectItem value="predicted2026">2026 Prediction</SelectItem>
              </SelectContent>
            </Select>
            <Select 
              value={sortBy} 
              onValueChange={setSortBy}
            >
              <SelectTrigger className="w-[160px]">
                <SelectValue placeholder="Sort By" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="decile">Sort by Decile</SelectItem>
                <SelectItem value="predicted">Sort by Premium</SelectItem>
                <SelectItem value="risk">Sort by Risk</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="flex-col space-y-8">
          <div className="flex items-center flex-wrap gap-4">
            <div className="flex items-center">
              <input
                type="checkbox"
                id="show-actual"
                checked={showActual}
                onChange={() => setShowActual(!showActual)}
                className="mr-2"
              />
              <label htmlFor="show-actual" className="text-sm font-medium">
                Compare to 2023 Actual
              </label>
            </div>
            
            {/* Legend */}
            <div className="ml-auto flex flex-wrap gap-3">
              {["Very Low", "Low", "Medium", "High", "Very High", "Extreme"].map((risk) => (
                <div key={risk} className="flex items-center">
                  <div 
                    className="w-3 h-3 rounded-full mr-1"
                    style={{ backgroundColor: getRiskColor(risk) }}
                  ></div>
                  <span className="text-xs">{risk}</span>
                </div>
              ))}
            </div>
          </div>
          
          <ChartContainer className="h-[400px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={sortedData}
                margin={{ top: 20, right: 30, left: 20, bottom: 40 }}
                barCategoryGap={5}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="decile" 
                  label={{ value: "Customer Risk Decile", position: "insideBottom", dy: 10 }}
                  tick={{ fontSize: 12 }}
                />
                <YAxis 
                  label={{ value: "Pure Premium ($)", angle: -90, position: "insideLeft", dy: 40 }}
                  tick={{ fontSize: 12 }}
                />
                <RechartsTooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ paddingTop: 20 }} />
                
                {/* Actual 2023 Premium */}
                {showActual && (
                  <Bar 
                    dataKey="actual2023" 
                    name="2023 Actual" 
                    fill="#64748b" 
                    opacity={0.7}
                  />
                )}
                
                {/* Predicted Premium */}
                <Bar 
                  dataKey={selectedYear} 
                  name={yearLabels[selectedYear]} 
                  fill={(data) => getRiskColor(data.risk)}
                >
                  <Label
                    position="top"
                    content={({ x, y, width, height, value }) => {
                      return (
                        <text
                          x={x + width / 2}
                          y={y - 6}
                          textAnchor="middle"
                          fontSize={10}
                          fill="currentColor"
                        >
                          ${value}
                        </text>
                      );
                    }}
                  />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </ChartContainer>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="border rounded-lg p-3">
              <div className="text-sm text-muted-foreground">Avg. Premium Increase</div>
              <div className="text-lg font-bold mt-1 text-foreground">
                +{(purePremiumByDecile.reduce((sum, item) => sum + (item.predicted2026 - item.actual2023), 0) / 
                  purePremiumByDecile.reduce((sum, item) => sum + item.actual2023, 0) * 100).toFixed(1)}%
              </div>
            </div>
            <div className="border rounded-lg p-3">
              <div className="text-sm text-muted-foreground">Highest Risk Segment</div>
              <div className="text-lg font-bold mt-1 text-foreground">
                {purePremiumByDecile[9].predicted2026 / purePremiumByDecile[0].predicted2026 > 8 ? 
                  `${(purePremiumByDecile[9].predicted2026 / purePremiumByDecile[0].predicted2026).toFixed(1)}x higher risk` : 
                  'Top 10% of customers'}
              </div>
            </div>
            <div className="border rounded-lg p-3">
              <div className="text-sm text-muted-foreground">Risk Segmentation</div>
              <div className="text-lg font-bold mt-1 text-foreground">
                10 segments
              </div>
            </div>
            <div className="border rounded-lg p-3">
              <div className="text-sm text-muted-foreground">Pricing Confidence</div>
              <div className="text-lg font-bold mt-1 text-foreground">
                94.8% accuracy
              </div>
            </div>
          </div>
        </div>
      </CardContent>
      <CardFooter className="flex justify-between border-t pt-4">
        <div>
          <p className="text-sm text-muted-foreground">
            Based on data from 185,000+ customers
          </p>
        </div>
        <div className="flex items-center">
          <Button size="sm" variant="outline" className="mr-2">
            <Share2 className="h-4 w-4 mr-1" />
            Share
          </Button>
          <Button size="sm" variant="default">
            <Download className="h-4 w-4 mr-1" />
            Export
          </Button>
        </div>
        </CardFooter>
      </Card>
    );
// Sample pure premium by decile data
const purePremiumByDecile = [
  { 
    decile: 1, 
    actual2023: 142, 
    predicted2024: 145, 
    predicted2025: 150, 
    predicted2026: 155, 
    risk: "Very Low",
    pricingStatus: "optimal",
    riskReduction: { potential: 8, savings: 12000, action: "Maintain current approach" },
    peerComparison: { position: "top20", averagePremium: 152 },
    riskTrend: "stable",
    optimizationImpact: { revenue: "+0.8%", retention: "+0.5%" },
    recommendation: "No change needed; risk is well-managed"
  },
  { 
    decile: 2, 
    actual2023: 165, 
    predicted2024: 168, 
    predicted2025: 174, 
    predicted2026: 182, 
    risk: "Low",
    pricingStatus: "optimal",
    riskReduction: { potential: 10, savings: 18000, action: "Basic risk awareness program" },
    peerComparison: { position: "top30", averagePremium: 174 },
    riskTrend: "stable",
    optimizationImpact: { revenue: "+1.2%", retention: "+0.2%" },
    recommendation: "Implement quarterly risk assessment for high-value accounts"
  },
  { 
    decile: 3, 
    actual2023: 188, 
    predicted2024: 192, 
    predicted2025: 200, 
    predicted2026: 206, 
    risk: "Low",
    pricingStatus: "optimal",
    riskReduction: { potential: 12, savings: 25000, action: "Customer education" },
    peerComparison: { position: "average", averagePremium: 195 },
    riskTrend: "worsening",
    optimizationImpact: { revenue: "+1.8%", retention: "-0.2%" },
    recommendation: "Implement customer education program to reduce preventable claims"
  },
  { 
    decile: 4, 
    actual2023: 210, 
    predicted2024: 215, 
    predicted2025: 222, 
    predicted2026: 231, 
    risk: "Low-Medium",
    pricingStatus: "optimal",
    riskReduction: { potential: 15, savings: 32000, action: "Conduct property inspections" },
    peerComparison: { position: "average", averagePremium: 228 },
    riskTrend: "worsening",
    optimizationImpact: { revenue: "+3.2%", retention: "-0.8%" },
    recommendation: "Offer risk assessment services to reduce potential claims"
  },
  {
    decile: 5, 
    actual2023: 256, 
    predicted2024: 262, 
    predicted2025: 278, 
    predicted2026: 288, 
    risk: "Medium",
    pricingStatus: "underpriced",
    riskReduction: { potential: 18, savings: 48000, action: "Implement risk mitigation technologies" },
    peerComparison: { position: "bottom40", averagePremium: 305 },
    riskTrend: "worsening",
    optimizationImpact: { revenue: "+6.5%", retention: "-1.2%" },
    recommendation: "Increase premiums by 6-8% and offer risk mitigation incentives"
  },
  {
    decile: 6, 
    actual2023: 320, 
    predicted2024: 318, 
    predicted2025: 335, 
    predicted2026: 352, 
    risk: "Medium",
    pricingStatus: "optimal",
    riskReduction: { potential: 22, savings: 65000, action: "Require safety equipment upgrades" },
    peerComparison: { position: "average", averagePremium: 347 },
    riskTrend: "improving",
    optimizationImpact: { revenue: "+4.1%", retention: "-0.9%" },
    recommendation: "Target for safety equipment upgrades with premium discounts"
  },
  {
    decile: 7, 
    actual2023: 387, 
    predicted2024: 395, 
    predicted2025: 412, 
    predicted2026: 428, 
    risk: "Medium-High",
    pricingStatus: "underpriced",
    riskReduction: { potential: 25, savings: 92000, action: "Implement comprehensive risk management" },
    peerComparison: { position: "bottom30", averagePremium: 412 },
    riskTrend: "worsening",
    optimizationImpact: { revenue: "+7.8%", retention: "-2.5%" },
    recommendation: "Increase premiums 8-10% with tailored risk management program"
  },
  {
    decile: 8, 
    actual2023: 452, 
    predicted2024: 465, 
    predicted2025: 488, 
    predicted2026: 515, 
    risk: "High",
    pricingStatus: "underpriced",
    riskReduction: { potential: 35, savings: 120000, action: "Enhanced loss control measures" },
    peerComparison: { position: "bottom20", averagePremium: 495 },
    riskTrend: "worsening",
    optimizationImpact: { revenue: "+9.2%", retention: "-3.8%" },
    recommendation: "Implement mandatory loss control and increase premiums by 10-15%"
  },
  {
    decile: 9, 
    actual2023: 558, 
    predicted2024: 575, 
    predicted2025: 610, 
    predicted2026: 645, 
    risk: "Very High",
    pricingStatus: "severely underpriced",
    riskReduction: { potential: 45, savings: 180000, action: "Mandatory risk mitigation" },
    peerComparison: { position: "bottom10", averagePremium: 625 },
    riskTrend: "worsening",
    optimizationImpact: { revenue: "+15.5%", retention: "-8.2%" },
    recommendation: "Significant premium increase with customized risk management plan"
  },
  {
    decile: 10, 
    actual2023: 842, 
    predicted2024: 880, 
    predicted2025: 945, 
    predicted2026: 1025, 
    risk: "Extreme",
    pricingStatus: "severely underpriced",
    riskReduction: { potential: 60, savings: 350000, action
            <SelectTrigger className="w-[150px]">
              <SelectValue placeholder="Product" />
            </SelectTrigger>
            <SelectContent>
              {products.map(product => (
                <SelectItem key={product} value={product}>{product}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          
          <Select value={dateRange} onValueChange={(value: any) => setDateRange(value)}>
            <SelectTrigger className="w-[150px]">
              <SelectValue placeholder="Date Range" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="1y">Last Year</SelectItem>
              <SelectItem value="3y">Last 3 Years</SelectItem>
              <SelectItem value="5y">Last 5 Years</SelectItem>
              <SelectItem value="10y">Last 10 Years</SelectItem>
            </SelectContent>
          </Select>
          
          <Select value={lineView} onValueChange={(value: any) => setLineView(value)}>
            <SelectTrigger className="w-[150px]">
              <SelectValue placeholder="View" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="monthly">Monthly</SelectItem>
              <SelectItem value="quarterly">Quarterly</SelectItem>
              <SelectItem value="annual">Annual</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>
      
      {/* Show error message if there is an error */}
      {error && (
        <Card className="bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800">
          <CardContent className="pt-6">
            <div className="flex items-center gap-2 text-yellow-800 dark:text-yellow-400">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-alert-triangle"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>
              <div>
                <p className="font-medium">Warning: Using Fallback Data</p>
                <p className="text-sm">{error} - Showing fallback data until API is available.</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
      
      {/* Loading state */}
      {isLoading ? (
        <div className="space-y-6">
          <Skeleton className="h-[400px] w-full rounded-lg" />
          <Skeleton className="h-[400px] w-full rounded-lg" />
        </div>
      ) : (
        <>
          <Tabs defaultValue="chart" className="w-full">
            <TabsList>
              <TabsTrigger value="chart">Charts</TabsTrigger>
              <TabsTrigger value="table">Data Table</TabsTrigger>
            </TabsList>
            <TabsContent value="chart" className="space-y-6">
              <HistoricalVsProjectedLosses />
              <PurePremiumByDecile />
            </TabsContent>
            <TabsContent value="table">
              <Card>
                <CardHeader>
                  <CardTitle>Raw Data</CardTitle>
                  <CardDescription>
                    The raw data used to generate the charts
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-[500px]">
                    <div className="space-y-8">
                      <div>
                        <h3 className="text-lg font-medium mb-2">Historical Data</h3>
                        <table className="w-full border-collapse">
                          <thead>
                            <tr className="border-b">
                              <th className="text-left p-2">Date</th>
                              <th className="text-right p-2">Value</th>
                            </tr>
                          </thead>
                          <tbody>
                            {historicalData.map((item, index) => (
                              <tr key={index} className="border-b">
                                <td className="p-2">{format(item.date, 'MMM dd, yyyy')}</td>
                                <td className="text-right p-2">{formatCurrency(item.value)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      
                      <div>
                        <h3 className="text-lg font-medium mb-2">Projected Data</h3>
                        <table className="w-full border-collapse">
                          <thead>
                            <tr className="border-b">
                              <th className="text-left p-2">Date</th>
                              <th className="text-right p-2">Value</th>
                              <th className="text-right p-2">Lower Bound</th>
                              <th className="text-right p-2">Upper Bound</th>
                            </tr>
                          </thead>
                          <tbody>
                            {projectedData.map((item, index) => {
                              const confidenceItem = confidenceInterval[index];
                              return (
                                <tr key={index} className="border-b">
                                  <td className="p-2">{format(item.date, 'MMM dd, yyyy')}</td>
                                  <td className="text-right p-2">{formatCurrency(item.value)}</td>
                                  <td className="text-right p-2">{formatCurrency(confidenceItem?.lower || 0)}</td>
                                  <td className="text-right p-2">{formatCurrency(confidenceItem?.upper || 0)}</td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                      
                      <div>
                        <h3 className="text-lg font-medium mb-2">Decile Data</h3>
                        <table className="w-full border-collapse">
                          <thead>
                            <tr className="border-b">
                              <th className="text-left p-2">Decile</th>
                              <th className="text-right p-2">Predicted</th>
                              <th className="text-right p-2">Actual</th>
                              <th className="text-right p-2">Difference</th>
                            </tr>
                <label htmlFor="compare-baseline" className="text-sm font-medium">
                  Compare to Baseline
                </label>
