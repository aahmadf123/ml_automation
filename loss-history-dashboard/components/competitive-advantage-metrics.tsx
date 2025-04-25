"use client";

import React, { useState } from "react";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ChartContainer } from "@/components/ui/chart";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Download, HelpCircle, Zap, Clock, BarChart3, Gauge } from "lucide-react";
import { format } from "date-fns";
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  LineChart,
  Line,
  Label,
  RadialBarChart,
  RadialBar,
  ReferenceLine,
} from "recharts";

// Sample data for Model Accuracy
const accuracyData = [
  { name: "Our ML Model", value: 97.8, fill: "#10b981" }, // Green
  { name: "Industry Average", value: 86.3, fill: "#f97316" }, // Orange
  { name: "Traditional Methods", value: 74.2, fill: "#ef4444" }, // Red
];

// Sample data for time-to-detection improvements
const detectionTimeData = [
  { year: 2021, mlModelDays: 32, traditionalDays: 64, improvement: "50%" },
  { year: 2022, mlModelDays: 24, traditionalDays: 58, improvement: "59%" },
  { year: 2023, mlModelDays: 18, traditionalDays: 55, improvement: "67%" },
  { year: 2024, mlModelDays: 12, traditionalDays: 52, improvement: "77%" },
  { year: 2025, mlModelDays: 8, traditionalDays: 50, improvement: "84%" },
];

// Sample data for pricing precision by segment
const pricingPrecisionData = [
  { segment: "Low Risk", ourError: 2.4, industryError: 7.8, improvement: "69%" },
  { segment: "Medium Risk", ourError: 3.6, industryError: 9.2, improvement: "61%" },
  { segment: "High Risk", ourError: 4.9, industryError: 13.4, improvement: "63%" },
  { segment: "Very High Risk", ourError: 6.5, industryError: 15.8, improvement: "59%" },
  { segment: "Extreme Risk", ourError: 7.2, industryError: 18.5, improvement: "61%" },
];

// Sample data for risk assessment speed
const assessmentSpeedData = [
  { name: "Our ML Model", value: 95, fill: "#10b981" }, // Percentile ranking (higher is better)
  { name: "Top Competitors", value: 82, fill: "#3b82f6" },
  { name: "Industry Median", value: 50, fill: "#f97316" },
  { name: "Traditional Methods", value: 30, fill: "#ef4444" },
];

// Sample data for time improvement
const timelineData = [
  { month: "Jan", time: 45 },
  { month: "Feb", time: 42 },
  { month: "Mar", time: 38 },
  { month: "Apr", time: 34 },
  { month: "May", time: 30 },
  { month: "Jun", time: 27 },
  { month: "Jul", time: 24 },
  { month: "Aug", time: 18 },
  { month: "Sep", time: 15 },
  { month: "Oct", time: 12 },
  { month: "Nov", time: 10 },
  { month: "Dec", time: 8 },
];

export function CompetitiveAdvantageMetrics() {
  const [activeTab, setActiveTab] = useState("accuracy");

  // Format percentage for display
  const formatPercent = (value: number) => {
    return `${value.toFixed(1)}%`;
  };

  const formatWithSign = (value: string) => {
    if (value.startsWith("-")) return value;
    return `+${value}`;
  };

  // Custom gauge chart for accuracy
  const renderGaugeChart = () => {
    const outerRadius = 140;
    const innerRadius = 100;

    return (
      <ChartContainer className="h-[400px]">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={[{ name: "Background", value: 100 }]}
              cx="50%"
              cy="50%"
              startAngle={180}
              endAngle={0}
              innerRadius={innerRadius}
              outerRadius={outerRadius}
              fill="#e5e7eb"
              stroke="none"
              paddingAngle={0}
              dataKey="value"
            />
            {accuracyData.map((entry, index) => (
              <Pie
                key={`gauge-${index}`}
                data={[{ name: entry.name, value: entry.value }, { name: "Remainder", value: 100 - entry.value }]}
                cx="50%"
                cy="50%"
                startAngle={180}
                endAngle={0}
                innerRadius={innerRadius - index * 20}
                outerRadius={outerRadius - index * 20}
                paddingAngle={0}
                dataKey="value"
                stroke="none"
              >
                <Cell key={`cell-${index}-0`} fill={entry.fill} />
                <Cell key={`cell-${index}-1`} fill="transparent" />
                <Label
                  value={`${entry.name}: ${entry.value}%`}
                  position="center"
                  fill="currentColor"
                  style={{ fontSize: 14, fontWeight: "bold", transform: `translateY(${index * 25 - 25}px)` }}
                />
              </Pie>
            ))}
          </PieChart>
        </ResponsiveContainer>
      </ChartContainer>
    );
  };

  // Timeline chart for detection time improvements
  const renderTimelineChart = () => {
    return (
      <ChartContainer className="h-[400px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={detectionTimeData}
            margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="year" />
            <YAxis 
              label={{ value: 'Days to Detect High-Risk Cases', angle: -90, position: 'insideLeft' }} 
              domain={[0, 'dataMax + 10']}
            />
            <RechartsTooltip 
              formatter={(value, name) => {
                if (name === "mlModelDays") return [`${value} days`, "ML Model"];
                if (name === "traditionalDays") return [`${value} days`, "Traditional Methods"];
                return [value, name];
              }}
            />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="traditionalDays" 
              name="Traditional Methods" 
              stroke="#ef4444" 
              strokeWidth={2} 
              dot={{ r: 5 }} 
            />
            <Line 
              type="monotone" 
              dataKey="mlModelDays" 
              name="ML Model" 
              stroke="#10b981" 
              strokeWidth={3} 
              dot={{ r: 6 }} 
            />
            {detectionTimeData.map((entry, index) => (
              <ReferenceLine 
                key={`ref-${index}`} 
                x={entry.year} 
                stroke="none"
                ifOverflow="extendDomain"
                label={{ 
                  value: entry.improvement, 
                  position: 'top', 
                  fill: '#10b981',
                  fontSize: 12,
                  fontWeight: 'bold'
                }}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </ChartContainer>
    );
  };

  // Bar chart for pricing precision
  const renderPricingPrecisionChart = () => {
    return (
      <ChartContainer className="h-[400px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={pricingPrecisionData}
            margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="segment" />
            <YAxis 
              label={{ value: 'Pricing Error (%)', angle: -90, position: 'insideLeft' }}
              domain={[0, 'dataMax + 5']}
            />
            <RechartsTooltip 
              formatter={(value, name) => {
                if (name === "ourError") return [`${value}%`, "Our ML Model"];
                if (name === "industryError") return [`${value}%`, "Industry Average"];
                return [value, name];
              }}
              labelFormatter={(value) => `Segment: ${value}`}
            />
            <Legend />
            <Bar 
              dataKey="industryError" 
              name="Industry Average" 
              fill="#f97316" 
              radius={[4, 4, 0, 0]}
            >
              <Label position="top" formatter={value => `${value}%`} />
            </Bar>
            <Bar 
              dataKey="ourError" 
              name="Our ML Model" 
              fill="#10b981" 
              radius={[4, 4, 0, 0]}
            >
              <Label position="top" formatter={value => `${value}%`} />
            </Bar>
            {pricingPrecisionData.map((entry, index) => (
              <ReferenceLine 
                key={`ref-${index}`} 
                x={entry.segment} 
                stroke="none"
                label={{ 
                  value: entry.improvement, 
                  position: 'top', 
                  fill: '#10b981',
                  fontSize: 12,
                  fontWeight: 'bold'
                }}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </ChartContainer>
    );
  };

  // Speedometer chart for risk assessment speed
  const renderSpeedometerChart = () => {
    return (
      <ChartContainer className="h-[400px]">
        <ResponsiveContainer width="100%" height="100%">
          <RadialBarChart 
            cx="50%" 
            cy="50%" 
            innerRadius="20%" 
            outerRadius="90%" 
            barSize={20} 
            data={assessmentSpeedData} 
            startAngle={180} 
            endAngle={0}
          >
            <RadialBar
              background
              clockWise
              dataKey="value"
              cornerRadius={10}
              label={{ 
                position: 'insideStart', 
                fill: '#fff', 
                fontWeight: 'bold', 
                formatter: (value) => `${value}%` 
              }}
            />
            <RechartsTooltip formatter={(value) => [`${value} percentile`, "Speed Rating"]} />
            <Legend 
              iconSize={10} 
              layout="vertical" 
              verticalAlign="middle" 
              align="right" 
              wrapperStyle={{ paddingLeft: 20 }}
            />
          </RadialBarChart>
        </ResponsiveContainer>
      </ChartContainer>
    );
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-xl font-semibold">Competitive Advantage Metrics</CardTitle>
            <CardDescription>
              How our ML-driven approach compares to industry standards
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
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
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="accuracy" className="flex items-center gap-2">
              <Gauge className="h-4 w-4" /> Model Accuracy
            </TabsTrigger>
            <TabsTrigger value="detection" className="flex items-center gap-2">
              <Clock className="h-4 w-4" /> Detection Speed
            </TabsTrigger>
            <TabsTrigger value="precision" className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4" /> Pricing Precision
            </TabsTrigger>
            <TabsTrigger value="speed" className="flex items-center gap-2">
              <Zap className="h-4 w-4" /> Assessment Speed
            </TabsTrigger>
          </TabsList>

          <div className="mt-6">
            <TabsContent value="accuracy" className="space-y-4">
              <div className="flex flex-col md:flex-row gap-6">
                <div className="md:w-1/3">
                  <div className="space-y-4">
                    <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
                      <h3 className="font-semibold text-lg text-green-800 dark:text-green-300">97.8% Model Accuracy</h3>
                      <p className="text-green-700 dark:text-green-400 text-sm mt-1">
                        Our ML models achieve industry-leading accuracy, outperforming traditional methods by 23.6 percentage points.
                      </p>
                    </div>
                    <div className="space-y-2">
                      <h4 className="font-medium">Key Improvements:</h4>
                      <ul className="space-y-1 text-sm">
                        <li className="flex items-start">
                          <span className="text-green-500 mr-2">✓</span>
                          <span>31.3% reduction in false positives compared to industry standard</span>
                        </li>
                        <li className="flex items-start">
                          <span className="text-green-500 mr-2">✓</span>
                          <span>42.7% reduction in false negatives for high-risk cases</span>
                        </li>
                        <li className="flex items-start">
                          <span className="text-green-500 mr-2">✓</span>
                          <span>67.8% higher consistency across different risk categories</span>
                        </li>
                        <li className="flex items-start">
                          <span className="text-green-500 mr-2">✓</span>
                          <span>Maintains 95%+ accuracy even with sparse historical data</span>
                        </li>
                      </ul>
                    </div>
                  </div>
                </div>
                <div className="md:w-2/3">{renderGaugeChart()}</div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
                <div className="border p-4 rounded-lg">
                  <div className="text-sm text-muted-foreground">Improvement vs Traditional</div>
                  <div className="text-2xl font-bold text-green-600">+23.6%</div>
                </div>
                <div className="border p-4 rounded-lg">
                  <div className="text-sm text-muted-foreground">False Positive Reduction</div>
                  <div className="text-2xl font-bold text-green-600">-31.3%</div>
                </div>
                <div className="border p-4 rounded-lg">
                  <div className="text-sm text-muted-foreground">False Negative Reduction</div>
                  <div className="text-2xl font-bold text-green-600">-42.7%</div>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="detection" className="space-y-4">
              <div className="flex flex-col md:flex-row gap-6">
                <div className="md:w-1/3">
                  <div className="space-y-4">
                    <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
                      <h3 className="font-semibold text-lg text-blue-800 dark:text-blue-300">84% Faster Detection</h3>
                      <p className="text-blue-700 dark:text-blue-400 text-sm mt-1">
                        Our ML-powered risk detection identifies high-risk cases 6.25x faster than traditional methods, reducing exposure time dramatically.
                      </p>
                    </div>
                    <div className="space-y-2">
                      <h4 className="font-medium">Key Benefits:</h4>
                      <ul className="space-y-1 text-sm">
                        <li className="flex items-start">
                          <span className="text-blue-500 mr-2">✓</span>
                          <span>Days instead of months to identify emerging risks</span>
                        </li>
                        <li className="flex items-start">
                          <span className="text-blue-500 mr-2">✓</span>
                          <span>Proactive interventions before claims escalate</span>
                        </li>
                        <li className="flex items-start">
                          <span className="text-blue-500 mr-2">✓</span>
                          <span>Automated alerting system for immediate action</span>
                        </li>
                        <li className="flex items-start">
                          <span className="text-blue-500 mr-2">✓</span>
                          <span>Annual detection improvement rate of 25%</span>
                        </li>
                      </ul>
                    </div>
                  </div>
                </div>
                <div className="md:w-2/3">{renderTimelineChart()}</div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
                <div className="border p-4 rounded-lg">
                  <div className="text-sm text-muted-foreground">Average Days to Detect</div>
                  <div className="text-2xl font-bold text-blue-600">8 days</div>
                  <div className="text-xs text-gray-500">vs. 50 days (traditional)</div>
                </div>
                <div className="border p-4 rounded-lg">
                  <div className="text-sm text-muted-foreground">Annual Improvement</div>
                  <div className="text-2xl font-bold text-blue-600">25%</div>
                </div>
                <div className="border p-4 rounded-lg">
                  <div className="text-sm text-muted-foreground">Detection Cost Savings</div>
                  <div className="text-2xl font-bold text-blue-600">$1.8M</div>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="precision" className="space-y-4">
              <div className="flex flex-col md:flex-row gap-6">
                <div className="md:w-1/3">
                  <div className="space-y-4">
                    <div className="bg-amber-50 dark:bg-amber-900/20 p-4 rounded-lg">
                      <h3 className="font-semibold text-lg text-amber-800 dark:text-amber-300">62.4% More Precise</h3>
                      <p className="text-amber-700 dark:text-amber-400 text-sm mt-1">
                        Our pricing models are 62.4% more precise than industry averages, particularly for high-risk segments.
                      </p>
                    </div>
                    <div className="space-y-2">
                      <h4 className="font-medium">Pricing Advantages:</h4>
                      <ul className="space-y-1 text-sm">
                        <li className="flex items-start">
                          <span className="text-amber-500 mr-2">✓</span>
                          <span>Granular pricing based on 150+ risk factors</span>
                        </li>
                        <li className="flex items-start">
                          <span className="text-amber-500 mr-2">✓</span>
                          <span>Dynamic adjustment based on real-time data</span>
                        </li>
                        <li className="flex items-start">
                          <span className="text-amber-500 mr-2">✓</span>
                          <span>Reduced pricing error in all risk segments</span>
                        </li>
                        <li className="flex items-start">
                          <span className="text-amber-500 mr-2">✓</span>
                          <span>Higher margin retention in competitive markets</span>
                        </li>
                      </ul>
                    </div>
                  </div>
                </div>
                <div className="md:w-2/3">{renderPricingPrecisionChart()}</div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
                <div className="border p-4 rounded-lg">
                  <div className="text-sm text-muted-foreground">Average Error Reduction</div>
                  <div className="text-2xl font-bold text-amber-600">62.4%</div>
                </div>
                <div className="border p-4 rounded-lg">
                  <div className="text-sm text-muted-foreground">Revenue Impact</div>
                  <div className="text-2xl font-bold text-amber-600">+$4.2M</div>
                </div>
                <div className="border p-4 rounded-lg">
                  <div className="text-sm text-muted-foreground">Retention Impact</div>
                  <div className="text-2xl font-bold text-amber-600">+2.8%</div>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="speed" className="space-y-4">
              <div className="flex flex-col md:flex-row gap-6">
                <div className="md:w-1/3">
                  <div className="space-y-4">
                    <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
                      <h3 className="font-semibold text-lg text-purple-800 dark:text-purple-300">95th Percentile Speed</h3>
                      <p className="text-purple-700 dark:text-purple-400 text-sm mt-1">
                        Our assessment speed ranks in the 95th percentile across the industry, enabling real-time risk evaluation and pricing.
                      </p>
                    </div>
                    <div className="space-y-2">
                      <h4 className="font-medium">Speed Benefits:</h4>
                      <ul className="space-y-1 text-sm">
                        <li className="flex items-start">
                          <span className="text-purple-500 mr-2">✓</span>
                          <span>Risk assessment in milliseconds vs. minutes</span>
                        </li>
                        <li className="flex items-start">
                          <span className="text-purple-500 mr-2">✓</span>
                          <span>Real-time quote adjustments during customer interactions</span>
                        </li>
                        <li className="flex items-start">
                          <span className="text-purple-500 mr-2">✓</span>
                          <span>Enables dynamic pricing at point-of-sale</span>
                        </li>
                        <li className="flex items-start">
                          <span className="text-purple-500 mr-2">✓</span>
                          <span>3.2x faster than top competitors</span>
                        </li>
                      </ul>
                    </div>
                  </div>
                </div>
                <div className="md:w-2/3">{renderSpeedometerChart()}</div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
                <div className="border p-4 rounded-lg">
                  <div className="text-sm text-muted-foreground">Average Response Time</div>
                  <div className="text-2xl font-bold text-purple-600">14ms</div>
                  <div className="text-xs text-gray-500">vs. 2830ms (industry)</div>
                </div>
                <div className="border p-4 rounded-lg">
                  <div className="text-sm text-muted-foreground">Quote Conversion Lift</div>
                  <div className="text-2xl font-bold text-purple-600">+12.3%</div>
                </div>
                <div className="border p-4 rounded-lg">
                  <div className="text-sm text-muted-foreground">API Capacity</div>
                  <div className="text-2xl font-bold text-purple-600">5.2M/day</div>
                </div>
              </div>
            </TabsContent>
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
          <span>Independent benchmarks by Insurance Analytics Quarterly</span>
        </div>
      </CardFooter>
    </Card>
  );
}
