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
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Calendar } from "@/components/ui/calendar"
import { format, subMonths, subYears, addMonths, parseISO } from "date-fns"
import dynamic from 'next/dynamic'

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false })

// Regions and products for filters
const regions = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]
const products = ["Auto", "Home", "Commercial", "Life", "Health"]

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

// Sample pure premium by decile data
const purePremiumByDecile = [
  { decile: 1, actual2023: 120, predicted2024: 118, predicted2025: 125, predicted2026: 129, risk: "Very Low" },
  { decile: 2, actual2023: 145, predicted2024: 147, predicted2025: 152, predicted2026: 156, risk: "Low" },
  { decile: 3, actual2023: 178, predicted2024: 175, predicted2025: 183, predicted2026: 188, risk: "Low" },
  { decile: 4, actual2023: 210, predicted2024: 215, predicted2025: 222, predicted2026: 231, risk: "Low-Medium" },
  { decile: 5, actual2023: 256, predicted2024: 262, predicted2025: 278, predicted2026: 288, risk: "Medium" },
  { decile: 6, actual2023: 320, predicted2024: 318, predicted2025: 335, predicted2026: 352, risk: "Medium" },
  { decile: 7, actual2023: 387, predicted2024: 395, predicted2025: 412, predicted2026: 428, risk: "Medium-High" },
  { decile: 8, actual2023: 475, predicted2024: 483, predicted2025: 510, predicted2026: 535, risk: "High" },
  { decile: 9, actual2023: 612, predicted2024: 625, predicted2025: 655, predicted2026: 688, risk: "Very High" },
  { decile: 10, actual2023: 890, predicted2024: 905, predicted2025: 965, predicted2026: 1020, risk: "Extreme" },
];

// Calculate some useful statistics for the dashboard
const lossAccuracy = historicalLossData
  .filter(d => d.actual && d.predicted)
  .map(d => Math.abs(1 - d.ratio || 0))
  .reduce((sum, val, _, arr) => sum + val / arr.length, 0);

const lossAccuracyFormatted = (100 - lossAccuracy * 100).toFixed(1) + '%';

// Component for visualization of historical vs projected losses
const HistoricalVsProjectedLosses = () => {
  const [forecastYears, setForecastYears] = useState<number>(3);
  const [showConfidenceInterval, setShowConfidenceInterval] = useState<boolean>(true);
  const [compareToBaseline, setCompareToBaseline] = useState<boolean>(true);
  const [date, setDate] = useState<Date | undefined>(new Date());

  // Confidence interval data (95% confidence)
  const confidenceIntervals = {
    // Year: [lower bound percentage, upper bound percentage]
    2025: [0.92, 1.08],
    2026: [0.89, 1.14],
    2027: [0.85, 1.22]
  };
  
  // Baseline is a simple linear projection without ML
  const baselineProjection = [
    { year: 2025, predicted: 19200000 },
    { year: 2026, predicted: 19800000 },
    { year: 2027, predicted: 20400000 },
  ];
  
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

// Component for Pure Premium Prediction by Decile
const PurePremiumByDecile = () => {
  const [selectedYear, setSelectedYear] = useState<string>("predicted2026");
  const [showActual, setShowActual] = useState<boolean>(true);
  const [sortBy, setSortBy] = useState<string>("decile");
  
  const yearLabels = {
    "actual2023": "2023 Actual",
    "predicted2024": "2024 Prediction",
    "predicted2025": "2025 Prediction",
    "predicted2026": "2026 Prediction"
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
  
  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload || !payload.length) return null;
    
    const data = payload[0].payload;
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
};

export default function BusinessInsights() {
  // Filter state
  const [lineView, setLineView] = useState<"monthly" | "quarterly" | "annual">("monthly")
  const [dateRange, setDateRange] = useState<"1y" | "3y" | "5y" | "10y">("3y")
  const [selectedDate, setSelectedDate] = useState<Date>(new Date())
  const [selectedRegion, setSelectedRegion] = useState<string>("Northeast")
  const [selectedProduct, setSelectedProduct] = useState<string>("Auto")

  // Chart data state
  const [historicalData, setHistoricalData] = useState<Array<{ date: Date; value: number }>>([])
  const [projectedData, setProjectedData] = useState<Array<{ date: Date; value: number }>>([])
  const [confidenceInterval, setConfidenceInterval] = useState<Array<{ date: Date; lower: number; upper: number }>>([])
  const [decileData, setDecileData] = useState<Array<{ decile: number; predicted: number; actual: number }>>([])
  
  // Loading and error states
  const [isLoading, setIsLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)

  // Fetch data from the API
  useEffect(() => {
    async function fetchProjectionsData() {
      setIsLoading(true);
      setError(null);
      
      try {
        // Construct the API URL with filters
        const apiUrl = `/api/projections?region=${selectedRegion}&product=${selectedProduct}`;
        console.log(`Fetching projections data from: ${apiUrl}`);
        
        const response = await fetch(apiUrl);
        
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.message || `API returned status ${response.status}`);
        }
        
        const result = await response.json();
        
        // Process the API response
        if (result && result.success && result.data) {
          const data = result.data;
          
          // Check if we have the expected data structure
          if (data.historical_data && data.projected_data && data.decile_data) {
            // Access the nested data for the selected region and product
            const historical = data.historical_data[selectedRegion]?.[selectedProduct] || [];
            const projected = data.projected_data[selectedRegion]?.[selectedProduct] || [];
            const deciles = data.decile_data[selectedRegion]?.[selectedProduct] || [];
            
            // Check if we actually have data to display
            if (historical.length === 0 && projected.length === 0) {
              console.warn(`No data found for ${selectedRegion}/${selectedProduct}`);
              throw new Error(`No data available for ${selectedRegion} - ${selectedProduct}`);
            }
            
            // Convert historical data
            const processedHistorical = historical.map((item: any) => ({
              date: parseISO(item.date),
              value: item.value
            }));
            
            // Convert projected data
            const processedProjected = projected.map((item: any) => ({
              date: parseISO(item.date),
              value: item.value
            }));
            
            // Convert confidence interval data
            const processedConfidence = projected.map((item: any) => ({
              date: parseISO(item.date),
              lower: item.lower,
              upper: item.upper
            }));
            
            // Convert decile data
            const processedDeciles = deciles.map((item: any) => ({
              decile: item.decile,
              predicted: item.predicted,
              actual: item.actual
            }));
            
            // Store the processed data in state
            setHistoricalData(processedHistorical);
            setProjectedData(processedProjected);
            setConfidenceInterval(processedConfidence);
            setDecileData(processedDeciles);
            
            console.log("Data processed successfully");
          } else {
            // Handle the case where the expected data structure is not present
            console.warn("Unexpected data structure in API response:", data);
            throw new Error("The data format from the API was not as expected");
          }
        } else {
          // Handle unsuccessful API response
          console.warn("API response was not successful:", result);
          throw new Error(result.error || "Failed to retrieve projections data");
        }
      } catch (err) {
        console.error("Error fetching projections data:", err);
        setError(err instanceof Error ? err.message : "An error occurred while fetching data");
        
        // Fall back to generating mock data
        generateMockData();
      } finally {
        setIsLoading(false);
      }
    }
    
    // Call the fetch function
    fetchProjectionsData();
    
  }, [dateRange, lineView, selectedRegion, selectedProduct]);
  
  // Helper function to aggregate data points
  function aggregateData(data: Array<{ date: Date; value: number }>, periodMonths: number) {
    const aggregated: Array<{ date: Date; value: number }> = []
    
    for (let i = 0; i < data.length; i += periodMonths) {
      const chunk = data.slice(i, i + periodMonths)
      if (chunk.length > 0) {
        // Use the last date in the period
        const periodDate = chunk[chunk.length - 1].date
        // Sum the values
        const periodValue = chunk.reduce((sum, point) => sum + point.value, 0)
        
        aggregated.push({
          date: new Date(periodDate),
          value: periodValue
        })
      }
    }
    
    return aggregated
  }

  return (
    <div className="space-y-8">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Business Insights</h2>
          <p className="text-muted-foreground">
            Analytics and historical/projected performance metrics
          </p>
        </div>
        
        <div className="flex flex-wrap items-center gap-2">
          <Select value={selectedRegion} onValueChange={setSelectedRegion}>
            <SelectTrigger className="w-[150px]">
              <SelectValue placeholder="Region" />
            </SelectTrigger>
            <SelectContent>
              {regions.map(region => (
                <SelectItem key={region} value={region}>{region}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          
          <Select value={selectedProduct} onValueChange={setSelectedProduct}>
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
                          </thead>
                          <tbody>
                            {decileData.map((item, index) => (
                              <tr key={index} className="border-b">
                                <td className="p-2">{item.decile}</td>
                                <td className="text-right p-2">{formatCurrency(item.predicted)}</td>
                                <td className="text-right p-2">{formatCurrency(item.actual)}</td>
                                <td className="text-right p-2">
                                  {(((item.actual / item.predicted) - 1) * 100).toFixed(1)}%
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </>
      )}
    </div>
  )
} 