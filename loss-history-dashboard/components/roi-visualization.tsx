"use client";

import React, { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ChartContainer } from "@/components/ui/chart";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { CalendarIcon, Download, HelpCircle, Info, DollarSign, TrendingUp, AlertCircle } from "lucide-react";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Calendar } from "@/components/ui/calendar";
import { format, parseISO } from "date-fns";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  ReferenceLine,
  ReferenceArea,
  Area,
  AreaChart,
  ComposedChart,
  LabelList,
} from "recharts";

// Sample ROI data
const costSavingsData = [
  { year: 2023, mlModelSavings: 1200000, traditionalCosts: 3600000 },
  { year: 2024, mlModelSavings: 1800000, traditionalCosts: 4200000 },
  { year: 2025, mlModelSavings: 2400000, traditionalCosts: 4900000 },
  { year: 2026, mlModelSavings: 3100000, traditionalCosts: 5700000 },
  { year: 2027, mlModelSavings: 3900000, traditionalCosts: 6600000 },
];

// Calculate savings percentage
const savingsData = costSavingsData.map(item => ({
  ...item,
  savingsPercentage: ((item.traditionalCosts - item.mlModelSavings) / item.traditionalCosts * 100).toFixed(1),
  totalSavings: item.traditionalCosts - item.mlModelSavings
}));

// Loss avoidance metrics
const lossAvoidanceData = [
  { year: 2023, quarter: "Q1", avoidedLosses: 320000, cumulativeAvoidedLosses: 320000 },
  { year: 2023, quarter: "Q2", avoidedLosses: 380000, cumulativeAvoidedLosses: 700000 },
  { year: 2023, quarter: "Q3", avoidedLosses: 410000, cumulativeAvoidedLosses: 1110000 },
  { year: 2023, quarter: "Q4", avoidedLosses: 490000, cumulativeAvoidedLosses: 1600000 },
  { year: 2024, quarter: "Q1", avoidedLosses: 510000, cumulativeAvoidedLosses: 2110000 },
  { year: 2024, quarter: "Q2", avoidedLosses: 580000, cumulativeAvoidedLosses: 2690000 },
  { year: 2024, quarter: "Q3", avoidedLosses: 620000, cumulativeAvoidedLosses: 3310000 },
  { year: 2024, quarter: "Q4", avoidedLosses: 680000, cumulativeAvoidedLosses: 3990000 },
];

// Premium optimization data (waterfall chart data)
const premiumOptimizationData = [
  { name: "Base Premium", value: 1000, step: "start" },
  { name: "Risk Assessment", value: 120, step: "positive" },
  { name: "Claims History", value: 80, step: "positive" },
  { name: "ML Risk Factor", value: 150, step: "positive" },
  { name: "Competitive Adj.", value: -50, step: "negative" },
  { name: "Loyalty Discount", value: -30, step: "negative" },
  { name: "Optimized Premium", value: 1270, step: "end" },
];

// 3-year ROI projection with confidence intervals
const roiProjectionData = [
  { year: 2023, roi: 115, lowerCI: 108, upperCI: 122 },
  { year: 2024, roi: 142, lowerCI: 132, upperCI: 152 },
  { year: 2025, roi: 168, lowerCI: 155, upperCI: 181 },
  { year: 2026, roi: 195, lowerCI: 178, upperCI: 212 },
  { year: 2027, roi: 228, lowerCI: 204, upperCI: 252 },
];

export function ROIVisualization() {
  const [activeTab, setActiveTab] = useState("costSavings");
  const [selectedYear, setSelectedYear] = useState<string>("2025");
  const [showConfidenceInterval, setShowConfidenceInterval] = useState<boolean>(true);
  const [date, setDate] = useState<Date | undefined>(new Date());
  const [isLoading, setIsLoading] = useState<boolean>(false);

  // Format currency for tooltips and axes
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  // Format currency in millions for axis labels
  const formatMillions = (value: number) => {
    return `$${(value / 1000000).toFixed(1)}M`;
  };

  // Fetch ROI data from API
  useEffect(() => {
    const fetchROIData = async () => {
      setIsLoading(true);
      try {
        // In a real implementation, this would be fetching from API
        // const response = await fetch('/api/projections?type=roi&year=' + selectedYear);
        // const data = await response.json();
        
        // Simulate API call delay
        await new Promise(resolve => setTimeout(resolve, 500));
        
        // Use the mock data for now
        setIsLoading(false);
      } catch (error) {
        console.error("Error fetching ROI data:", error);
        setIsLoading(false);
      }
    };

    fetchROIData();
  }, [selectedYear]);

  // Custom tooltip for the cost savings chart
  const CostSavingsTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload || !payload.length) return null;
    
    const data = payload[0].payload;
    
    return (
      <div className="bg-background border rounded-lg shadow-lg p-3 text-sm">
        <h3 className="font-medium">{label}</h3>
        <div className="mt-2 space-y-1">
          <div className="flex items-center">
            <div className="w-3 h-3 rounded-full mr-2" style={{ backgroundColor: "#10b981" }}></div>
            <span className="font-medium">ML Model Cost:</span>
            <span className="ml-2">{formatCurrency(data.mlModelSavings)}</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 rounded-full mr-2" style={{ backgroundColor: "#f97316" }}></div>
            <span className="font-medium">Traditional Cost:</span>
            <span className="ml-2">{formatCurrency(data.traditionalCosts)}</span>
          </div>
          <div className="pt-1 border-t mt-1">
            <span className="font-medium">Total Savings:</span>
            <span className="ml-2 text-emerald-600">{formatCurrency(data.totalSavings)}</span>
          </div>
          <div>
            <span className="font-medium">Savings Rate:</span>
            <span className="ml-2 text-emerald-600">{data.savingsPercentage}%</span>
          </div>
        </div>
      </div>
    );
  };

  // Custom tooltip for the loss avoidance chart
  const LossAvoidanceTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload || !payload.length) return null;
    
    const data = payload[0].payload;
    
    return (
      <div className="bg-background border rounded-lg shadow-lg p-3 text-sm">
        <h3 className="font-medium">{data.year} {data.quarter}</h3>
        <div className="mt-2 space-y-1">
          <div className="flex items-center">
            <div className="w-3 h-3 rounded-full mr-2" style={{ backgroundColor: "#3b82f6" }}></div>
            <span className="font-medium">Avoided Losses:</span>
            <span className="ml-2">{formatCurrency(data.avoidedLosses)}</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 rounded-full mr-2" style={{ backgroundColor: "#10b981" }}></div>
            <span className="font-medium">Cumulative Avoided:</span>
            <span className="ml-2">{formatCurrency(data.cumulativeAvoidedLosses)}</span>
          </div>
        </div>
      </div>
    );
  };

  // Custom tooltip for the premium optimization chart
  const PremiumOptimizationTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload || !payload.length) return null;
    
    const data = payload[0].payload;
    
    return (
      <div className="bg-background border rounded-lg shadow-lg p-3 text-sm">
        <h3 className="font-medium">{data.name}</h3>
        <div className="mt-2">
          <div className="flex items-center">
            <span className="font-medium">Impact:</span>
            <span className={`ml-2 ${data.step === "negative" ? "text-red-500" : data.step === "positive" ? "text-emerald-600" : ""}`}>
              {data.step !== "start" && data.step !== "end" ? (data.value > 0 ? "+" : "") : ""}
              {formatCurrency(data.value)}
            </span>
          </div>
        </div>
      </div>
    );
  };

  // Custom tooltip for the ROI projection chart
  const ROIProjectionTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload || !payload.length) return null;
    
    const data = payload[0].payload;
    
    return (
      <div className="bg-background border rounded-lg shadow-lg p-3 text-sm">
        <h3 className="font-medium">{label}</h3>
        <div className="mt-2 space-y-1">
          <div className="flex items-center">
            <div className="w-3 h-3 rounded-full mr-2" style={{ backgroundColor: "#10b981" }}></div>
            <span className="font-medium">ROI:</span>
            <span className="ml-2">{data.roi}%</span>
          </div>
          {showConfidenceInterval && (
            <div className="pt-1 border-t mt-1">
              <span className="font-medium">95% Confidence Interval:</span>
              <div className="ml-2">{data.lowerCI}% - {data.upperCI}%</div>
            </div>
          )}
        </div>
      </div>
    );
  };

  // Render different charts based on active tab
  const renderChart = () => {
    if (isLoading) {
      return <Skeleton className="h-[400px] w-full" />;
    }

    switch (activeTab) {
      case "costSavings":
        return (
          <ChartContainer className="h-[400px]">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart
                data={savingsData}
                margin={{ top: 20, right: 30, left: 20, bottom: 10 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="year" />
                <YAxis 
                  yAxisId="left"
                  tickFormatter={formatMillions}
                />
                <YAxis 
                  yAxisId="right"
                  orientation="right"
                  tickFormatter={(value) => `${value}%`}
                />
                <RechartsTooltip content={<CostSavingsTooltip />} />
                <Legend />
                
                <Bar 
                  yAxisId="left"
                  dataKey="mlModelSavings" 
                  name="ML Model Cost" 
                  fill="#10b981" 
                  barSize={30}
                  radius={[4, 4, 0, 0]}
                >
                  <LabelList dataKey="savingsPercentage" position="top" formatter={(value) => `${value}%`} />
                </Bar>
                <Bar 
                  yAxisId="left"
                  dataKey="traditionalCosts" 
                  name="Traditional Cost" 
                  fill="#f97316" 
                  barSize={30}
                  radius={[4, 4, 0, 0]}
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="savingsPercentage"
                  name="Savings %"
                  stroke="#4f46e5"
                  strokeWidth={2}
                  dot={{ r: 5 }}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </ChartContainer>
        );
      
      case "lossAvoidance":
        return (
          <ChartContainer className="h-[400px]">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart
                data={lossAvoidanceData}
                margin={{ top: 20, right: 30, left: 20, bottom: 10 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="quarter" tickFormatter={(quarter, index) => `${lossAvoidanceData[index].year} ${quarter}`} />
                <YAxis 
                  yAxisId="left"
                  tickFormatter={formatMillions}
                />
                <YAxis 
                  yAxisId="right"
                  orientation="right"
                  tickFormatter={formatMillions}
                />
                <RechartsTooltip content={<LossAvoidanceTooltip />} />
                <Legend />
                
                <Bar 
                  yAxisId="left"
                  dataKey="avoidedLosses" 
                  name="Quarterly Avoided Losses" 
                  fill="#3b82f6" 
                  barSize={30}
                  radius={[4, 4, 0, 0]}
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="cumulativeAvoidedLosses"
                  name="Cumulative Avoided Losses"
                  stroke="#10b981"
                  strokeWidth={2}
                  dot={{ r: 5 }}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </ChartContainer>
        );
      
      case "premiumOptimization":
        // Create a custom waterfall chart for premium optimization
        return (
          <ChartContainer className="h-[400px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={premiumOptimizationData}
                margin={{ top: 20, right: 30, left: 20, bottom: 40 }}
                barCategoryGap={10}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="name" 
                  angle={-45} 
                  textAnchor="end" 
                  height={80}
                />
                <YAxis 
                  tickFormatter={(value) => formatCurrency(value)}
                />
                <RechartsTooltip content={<PremiumOptimizationTooltip />} />
                <Legend />
                <Bar
                  dataKey="value"
                  name="Premium Component"
                  fill={(data) => {
                    if (data.step === "start" || data.step === "end") return "#4f46e5";
                    return data.step === "positive" ? "#10b981" : "#ef4444";
                  }}
                  radius={[4, 4, 0, 0]}
                >
                  <LabelList 
                    dataKey="value" 
                    position="top" 
                    formatter={(value, entry) => {
                      if (entry.step === "start" || entry.step === "end") return formatCurrency(value);
                      return (value > 0 ? "+" : "") + formatCurrency(value);
                    }}
                    fill={(entry) => {
                      if (entry.step === "start" || entry.step === "end") return "#4f46e5";
                      return entry.step === "positive" ? "#10b981" : "#ef4444";
                    }}
                  />
                </Bar>
                <ReferenceLine y={0} stroke="#000" />
              </BarChart>
            </ResponsiveContainer>
          </ChartContainer>
        );

      case "roiProjection":
        return (
          <ChartContainer className="h-[400px]">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart
                data={roiProjectionData}
                margin={{ top: 20, right: 30, left: 20, bottom: 10 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="year" />
                <YAxis 
                  domain={[100, 'dataMax + 20']}
                  tickFormatter={(value) => `${value}%`}
                />
                <RechartsTooltip content={<ROIProjectionTooltip />} />
                <Legend />
                
                <Line 
                  type="monotone" 
                  dataKey="roi" 
                  name="Return on Investment" 
                  stroke="#10b981" 
                  strokeWidth={3} 
                  dot={{ r: 6 }}
                  isAnimationActive={true}
                />
                
                {/* Confidence Intervals */}
                {showConfidenceInterval && roiProjectionData.map((point) => (
                  <ReferenceArea
                    key={point.year}
                    x1={point.year - 0.4}
                    x2={point.year + 0.4}
                    y1={point.lowerCI}
                    y2={point.upperCI}
                    fill="#10b981"
                    fillOpacity={0.1}
                    ifOverflow="extendDomain"
                  />
                ))}
              </ComposedChart>
            </ResponsiveContainer>
          </ChartContainer>
        );

      default:
        return (
          <div className="flex items-center justify-center h-[400px] text-muted-foreground">
            <div className="text-center">
              <p>Select a chart type to visualize ROI data</p>
            </div>
          </div>
        );
    }
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-xl font-semibold">ROI & Financial Impact</CardTitle>
            <CardDescription>
              Quantified business value and return on investment metrics
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Popover>
              <PopoverTrigger asChild>
                <Button
                  variant={"outline"}
                  className="w-[200px] justify-start text-left font-normal"
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
            <Select value={selectedYear} onValueChange={setSelectedYear}>
              <SelectTrigger className="w-[120px]">
                <SelectValue placeholder="Year" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="2025">2025</SelectItem>
                <SelectItem value="2026">2026</SelectItem>
                <SelectItem value="2027">2027</SelectItem>
              </SelectContent>
            </Select>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="outline" size="icon">
                    <Download className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Download Report</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="costSavings">Cost Savings</TabsTrigger>
            <TabsTrigger value="lossAvoidance">Loss Avoidance</TabsTrigger>
            <TabsTrigger value="premiumOptimization">Premium Optimization</TabsTrigger>
            <TabsTrigger value="roiProjection">ROI Projection</TabsTrigger>
          </TabsList>
          <div className="mt-4">
            <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4">
              <div className="flex gap-4 flex-wrap">
                {activeTab === "roiProjection" && (
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
                )}
              </div>
              <div className="flex items-center">
                <div className="bg-green-50 dark:bg-green-950 px-3 py-1 rounded-md mr-3">
                  <span className="text-sm font-bold text-green-700 dark:text-green-300">
                    {activeTab === "costSavings" && "Avg. Savings: 33.2%"}
                    {activeTab === "lossAvoidance" && "Total Avoided: $3.99M"}
                    {activeTab === "premiumOptimization" && "Optimization: +27%"}
                    {activeTab === "roiProjection" && "3-Year ROI: 195%"}
                  </span>
                </div>
                <Badge variant="outline" className="bg-blue-50 dark:bg-blue-950 text-blue-700 dark:text-blue-300">
                  Last Updated: {format(new Date(), "MMM d, yyyy")}
                </Badge>
              </div>
            </div>
            
            {renderChart()}
          </div>
        </Tabs>
      </CardContent>
      <CardFooter className="flex justify-between border-t pt-4">
        <div>
          <p className="text-sm text-muted-foreground">
            Data as of {format(new Date(), "MMMM d, yyyy")}
          </p>
        </div>
        <div className="flex items-center text-sm text-muted-foreground">
          <HelpCircle className="h-4 w-4 mr-1" />
          <span>ROI calculations based on ML model accuracy of 97.8%</span>
        </div>
      </CardFooter>
    </Card>
  );
}
