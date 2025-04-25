"use client";

import React, { useState } from "react";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ChartContainer } from "@/components/ui/chart";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Download, HelpCircle, LineChart, Activity, CheckCircle2, AlertCircle } from "lucide-react";
import { format } from "date-fns";
import {
  ResponsiveContainer,
  LineChart as RechartsLineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  Area,
  AreaChart,
  ReferenceLine,
  ComposedChart,
  Scatter,
  ScatterChart,
  ZAxis,
  Cell,
} from "recharts";

// Sample historical accuracy data
const accuracyTrendData = [
  { month: "Jan 22", accuracy: 95.2, industry: 85.4 },
  { month: "Feb 22", accuracy: 95.4, industry: 85.3 },
  { month: "Mar 22", accuracy: 95.6, industry: 85.6 },
  { month: "Apr 22", accuracy: 95.9, industry: 85.8 },
  { month: "May 22", accuracy: 96.1, industry: 85.7 },
  { month: "Jun 22", accuracy: 96.3, industry: 85.9 },
  { month: "Jul 22", accuracy: 96.5, industry: 86.0 },
  { month: "Aug 22", accuracy: 96.7, industry: 86.1 },
  { month: "Sep 22", accuracy: 96.9, industry: 86.0 },
  { month: "Oct 22", accuracy: 97.1, industry: 86.2 },
  { month: "Nov 22", accuracy: 97.3, industry: 86.3 },
  { month: "Dec 22", accuracy: 97.4, industry: 86.3 },
  { month: "Jan 23", accuracy: 97.5, industry: 86.4 },
  { month: "Feb 23", accuracy: 97.6, industry: 86.4 },
  { month: "Mar 23", accuracy: 97.6, industry: 86.3 },
  { month: "Apr 23", accuracy: 97.7, industry: 86.4 },
  { month: "May 23", accuracy: 97.7, industry: 86.3 },
  { month: "Jun 23", accuracy: 97.7, industry: 86.2 },
  { month: "Jul 23", accuracy: 97.8, industry: 86.3 },
  { month: "Aug 23", accuracy: 97.8, industry: 86.3 },
  { month: "Sep 23", accuracy: 97.8, industry: 86.3 },
  { month: "Oct 23", accuracy: 97.8, industry: 86.3 },
  { month: "Nov 23", accuracy: 97.8, industry: 86.3 },
  { month: "Dec 23", accuracy: 97.8, industry: 86.3 },
];

// Sample confidence interval coverage data
const confidenceIntervalData = [
  { month: "Jan 22", coverage: 92.1, expected: 95 },
  { month: "Feb 22", coverage: 92.4, expected: 95 },
  { month: "Mar 22", coverage: 92.8, expected: 95 },
  { month: "Apr 22", coverage: 93.2, expected: 95 },
  { month: "May 22", coverage: 93.5, expected: 95 },
  { month: "Jun 22", coverage: 93.8, expected: 95 },
  { month: "Jul 22", coverage: 94.1, expected: 95 },
  { month: "Aug 22", coverage: 94.3, expected: 95 },
  { month: "Sep 22", coverage: 94.5, expected: 95 },
  { month: "Oct 22", coverage: 94.7, expected: 95 },
  { month: "Nov 22", coverage: 94.9, expected: 95 },
  { month: "Dec 22", coverage: 95.0, expected: 95 },
  { month: "Jan 23", coverage: 95.1, expected: 95 },
  { month: "Feb 23", coverage: 95.2, expected: 95 },
  { month: "Mar 23", coverage: 95.1, expected: 95 },
  { month: "Apr 23", coverage: 95.1, expected: 95 },
  { month: "May 23", coverage: 95.2, expected: 95 },
  { month: "Jun 23", coverage: 95.1, expected: 95 },
  { month: "Jul 23", coverage: 95.0, expected: 95 },
  { month: "Aug 23", coverage: 95.1, expected: 95 },
  { month: "Sep 23", coverage: 95.0, expected: 95 },
  { month: "Oct 23", coverage: 95.1, expected: 95 },
  { month: "Nov 23", coverage: 95.0, expected: 95 },
  { month: "Dec 23", coverage: 95.0, expected: 95 },
];

// Sample model stability data (jitter/drift over time)
const stabilityData = [
  { x: 1, y: 0.02, model: "Auto", size: 30 },
  { x: 2, y: 0.03, model: "Auto", size: 30 },
  { x: 3, y: 0.01, model: "Auto", size: 30 },
  { x: 4, y: 0.02, model: "Auto", size: 30 },
  { x: 5, y: 0.01, model: "Auto", size: 30 },
  { x: 6, y: 0.02, model: "Auto", size: 30 },
  { x: 7, y: 0.01, model: "Auto", size: 30 },
  { x: 8, y: 0.01, model: "Auto", size: 30 },
  { x: 1, y: 0.03, model: "Home", size: 40 },
  { x: 2, y: 0.04, model: "Home", size: 40 },
  { x: 3, y: 0.03, model: "Home", size: 40 },
  { x: 4, y: 0.03, model: "Home", size: 40 },
  { x: 5, y: 0.03, model: "Home", size: 40 },
  { x: 6, y: 0.02, model: "Home", size: 40 },
  { x: 7, y: 0.02, model: "Home", size: 40 },
  { x: 8, y: 0.02, model: "Home", size: 40 },
  { x: 1, y: 0.05, model: "Commercial", size: 35 },
  { x: 2, y: 0.06, model: "Commercial", size: 35 },
  { x: 3, y: 0.04, model: "Commercial", size: 35 },
  { x: 4, y: 0.05, model: "Commercial", size: 35 },
  { x: 5, y: 0.04, model: "Commercial", size: 35 },
  { x: 6, y: 0.04, model: "Commercial", size: 35 },
  { x: 7, y: 0.03, model: "Commercial", size: 35 },
  { x: 8, y: 0.03, model: "Commercial", size: 35 },
];

// Sample actual vs predicted loss ratio data
const lossRatioData = [
  { quarter: "Q1 22", actual: 65.2, predicted: 64.5, difference: 0.7 },
  { quarter: "Q2 22", actual: 64.8, predicted: 64.0, difference: 0.8 },
  { quarter: "Q3 22", actual: 63.5, predicted: 63.1, difference: 0.4 },
  { quarter: "Q4 22", actual: 62.9, predicted: 62.7, difference: 0.2 },
  { quarter: "Q1 23", actual: 62.5, predicted: 62.3, difference: 0.2 },
  { quarter: "Q2 23", actual: 61.8, predicted: 61.7, difference: 0.1 },
  { quarter: "Q3 23", actual: 61.2, predicted: 61.1, difference: 0.1 },
  { quarter: "Q4 23", actual: 60.8, predicted: 60.7, difference: 0.1 },
];

export function HistoricalPerformance() {
  const [activeTab, setActiveTab] = useState("accuracy");

  // Format percentage for display
  const formatPercent = (value: number) => {
    return `${value.toFixed(1)}%`;
  };

  // Accuracy trend chart
  const renderAccuracyTrendChart = () => {
    return (
      <ChartContainer className="h-[400px]">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={accuracyTrendData}
            margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="month" 
              interval={3}
            />
            <YAxis 
              domain={[80, 100]} 
              tickFormatter={(value) => `${value}%`}
              label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft' }}
            />
            <RechartsTooltip 
              formatter={(value) => [`${value}%`, ""]}
            />
            <Legend />
            <Area 
              type="monotone" 
              dataKey="accuracy" 
              name="Our ML Model" 
              stroke="#10b981" 
              fill="#10b98120" 
              activeDot={{ r: 6 }}
            />
            <Area 
              type="monotone" 
              dataKey="industry" 
              name="Industry Average" 
              stroke="#f97316" 
              fill="#f9731620" 
              activeDot={{ r: 6 }}
            />
            <ReferenceLine 
              y={95} 
              stroke="#6b7280" 
              strokeDasharray="3 3"
              label={{ value: "Acceptable Threshold", position: "right", fill: "#6b7280" }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </ChartContainer>
    );
  };

  // Confidence interval coverage chart
  const renderConfidenceIntervalChart = () => {
    return (
      <ChartContainer className="h-[400px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={confidenceIntervalData}
            margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="month" 
              interval={3}
            />
            <YAxis 
              domain={[90, 100]} 
              tickFormatter={(value) => `${value}%`}
              label={{ value: 'CI Coverage (%)', angle: -90, position: 'insideLeft' }}
            />
            <RechartsTooltip 
              formatter={(value) => [`${value}%`, ""]}
            />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="coverage" 
              name="Actual Coverage" 
              stroke="#3b82f6" 
              strokeWidth={2} 
              dot={{ r: 4 }}
            />
            <Line 
              type="monotone" 
              dataKey="expected" 
              name="Expected Coverage (95%)" 
              stroke="#6b7280" 
              strokeWidth={2} 
              strokeDasharray="3 3"
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </ChartContainer>
    );
  };

  // Model stability chart
  const renderStabilityChart = () => {
    return (
      <ChartContainer className="h-[400px]">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart
            margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              type="number" 
              dataKey="x" 
              name="Time Period"
              label={{ value: 'Time Period (Quarters)', position: 'bottom' }}
              domain={[0, 9]} 
              tickCount={9}
            />
            <YAxis 
              type="number" 
              dataKey="y" 
              name="Prediction Jitter"
              label={{ value: 'Prediction Jitter (%)', angle: -90, position: 'insideLeft' }}
              domain={[0, 0.07]} 
              tickFormatter={(value) => `${(value * 100).toFixed(1)}%`}
            />
            <ZAxis type="number" dataKey="size" range={[60, 200]} />
            <RechartsTooltip 
              cursor={{ strokeDasharray: '3 3' }}
              formatter={(value, name, props) => {
                if (name === "Prediction Jitter") return [`${(Number(value) * 100).toFixed(2)}%`, name];
                return [value, name];
              }}
            />
            <Legend />
            <Scatter 
              name="Auto" 
              data={stabilityData.filter(d => d.model === "Auto")} 
              fill="#3b82f6" 
              line={{ stroke: '#3b82f6', strokeWidth: 1 }}
              shape="circle"
            />
            <Scatter 
              name="Home" 
              data={stabilityData.filter(d => d.model === "Home")} 
              fill="#10b981" 
              line={{ stroke: '#10b981', strokeWidth: 1 }}
              

