"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { AlertTriangle, Info } from "lucide-react"
import type { ThresholdConfig } from "./threshold-manager"
import { Line, LineChart, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer } from "recharts"

// Mock data for metrics over time
const metricHistoryData = {
  drift_property_value: [
    { date: "2023-04-01", value: 1.2 },
    { date: "2023-04-02", value: 1.5 },
    { date: "2023-04-03", value: 1.8 },
    { date: "2023-04-04", value: 2.1 },
    { date: "2023-04-05", value: 2.3 },
    { date: "2023-04-06", value: 2.7 },
    { date: "2023-04-07", value: 3.0 },
    { date: "2023-04-08", value: 3.2 },
    { date: "2023-04-09", value: 3.5 },
    { date: "2023-04-10", value: 3.8 },
    { date: "2023-04-11", value: 4.1 },
    { date: "2023-04-12", value: 4.5 },
    { date: "2023-04-13", value: 4.8 },
    { date: "2023-04-14", value: 5.2 },
    { date: "2023-04-15", value: 5.5 },
  ],
  drift_claim_amount: [
    { date: "2023-04-01", value: 4.2 },
    { date: "2023-04-02", value: 4.1 },
    { date: "2023-04-03", value: 4.0 },
    { date: "2023-04-04", value: 3.9 },
    { date: "2023-04-05", value: 3.8 },
    { date: "2023-04-06", value: 3.7 },
    { date: "2023-04-07", value: 3.6 },
    { date: "2023-04-08", value: 3.5 },
    { date: "2023-04-09", value: 3.6 },
    { date: "2023-04-10", value: 3.7 },
    { date: "2023-04-11", value: 3.8 },
    { date: "2023-04-12", value: 3.7 },
    { date: "2023-04-13", value: 3.6 },
    { date: "2023-04-14", value: 3.7 },
    { date: "2023-04-15", value: 3.7 },
  ],
  perf_rmse: [
    { date: "2023-04-01", value: 0.48 },
    { date: "2023-04-02", value: 0.47 },
    { date: "2023-04-03", value: 0.46 },
    { date: "2023-04-04", value: 0.45 },
    { date: "2023-04-05", value: 0.44 },
    { date: "2023-04-06", value: 0.43 },
    { date: "2023-04-07", value: 0.42 },
    { date: "2023-04-08", value: 0.41 },
    { date: "2023-04-09", value: 0.42 },
    { date: "2023-04-10", value: 0.43 },
    { date: "2023-04-11", value: 0.44 },
    { date: "2023-04-12", value: 0.43 },
    { date: "2023-04-13", value: 0.42 },
    { date: "2023-04-14", value: 0.41 },
    { date: "2023-04-15", value: 0.42 },
  ],
  perf_r2: [
    { date: "2023-04-01", value: 0.82 },
    { date: "2023-04-02", value: 0.83 },
    { date: "2023-04-03", value: 0.84 },
    { date: "2023-04-04", value: 0.85 },
    { date: "2023-04-05", value: 0.86 },
    { date: "2023-04-06", value: 0.87 },
    { date: "2023-04-07", value: 0.88 },
    { date: "2023-04-08", value: 0.87 },
    { date: "2023-04-09", value: 0.86 },
    { date: "2023-04-10", value: 0.85 },
    { date: "2023-04-11", value: 0.84 },
    { date: "2023-04-12", value: 0.85 },
    { date: "2023-04-13", value: 0.86 },
    { date: "2023-04-14", value: 0.87 },
    { date: "2023-04-15", value: 0.87 },
  ],
}

// Mock thresholds
const mockThresholds: ThresholdConfig[] = [
  {
    id: "1",
    name: "High Property Value Drift",
    metricType: "drift",
    metricName: "drift_property_value",
    operator: ">",
    value: 5,
    severity: "warning",
    enabled: true,
    notificationChannels: ["email", "dashboard"],
    createdAt: "2023-04-15T09:30:00",
    updatedAt: "2023-04-15T09:30:00",
  },
  {
    id: "2",
    name: "Critical RMSE Alert",
    metricType: "performance",
    metricName: "perf_rmse",
    operator: ">",
    value: 0.5,
    severity: "critical",
    enabled: true,
    notificationChannels: ["email", "slack", "dashboard"],
    createdAt: "2023-04-10T14:20:00",
    updatedAt: "2023-04-12T11:15:00",
  },
  {
    id: "3",
    name: "Low R² Score",
    metricType: "performance",
    metricName: "perf_r2",
    operator: "<",
    value: 0.8,
    severity: "info",
    enabled: false,
    notificationChannels: ["dashboard"],
    createdAt: "2023-04-05T16:45:00",
    updatedAt: "2023-04-05T16:45:00",
  },
]

// Available metrics for selection
const availableMetrics = {
  drift: [
    { id: "drift_property_value", name: "Property Value Drift" },
    { id: "drift_claim_amount", name: "Claim Amount Drift" },
  ],
  performance: [
    { id: "perf_rmse", name: "RMSE" },
    { id: "perf_r2", name: "R² Score" },
  ],
}

export function ThresholdVisualizer() {
  const [activeTab, setActiveTab] = useState<"drift" | "performance">("drift")
  const [selectedMetric, setSelectedMetric] = useState<string>("drift_property_value")
  const [thresholds, setThresholds] = useState<ThresholdConfig[]>(mockThresholds)
  const [metricData, setMetricData] = useState<any[]>([])
  const [activeThresholds, setActiveThresholds] = useState<ThresholdConfig[]>([])

  useEffect(() => {
    // Set default selected metric when tab changes
    if (activeTab === "drift") {
      setSelectedMetric("drift_property_value")
    } else {
      setSelectedMetric("perf_rmse")
    }
  }, [activeTab])

  useEffect(() => {
    // Update metric data when selected metric changes
    if (selectedMetric && metricHistoryData[selectedMetric as keyof typeof metricHistoryData]) {
      setMetricData(metricHistoryData[selectedMetric as keyof typeof metricHistoryData])
    } else {
      setMetricData([])
    }

    // Filter thresholds for the selected metric
    const filteredThresholds = thresholds.filter(
      (threshold) => threshold.metricName === selectedMetric && threshold.enabled,
    )
    setActiveThresholds(filteredThresholds)
  }, [selectedMetric, thresholds])

  const getThresholdColor = (severity: string) => {
    switch (severity) {
      case "info":
        return "#3b82f6" // blue
      case "warning":
        return "#f59e0b" // amber
      case "critical":
        return "#ef4444" // red
      default:
        return "#6b7280" // gray
    }
  }

  const getOperatorSymbol = (operator: string) => {
    switch (operator) {
      case ">":
        return ">"
      case "<":
        return "<"
      case ">=":
        return "≥"
      case "<=":
        return "≤"
      case "=":
        return "="
      default:
        return operator
    }
  }

  const checkThresholdViolation = (value: number, threshold: ThresholdConfig) => {
    switch (threshold.operator) {
      case ">":
        return value > threshold.value
      case "<":
        return value < threshold.value
      case ">=":
        return value >= threshold.value
      case "<=":
        return value <= threshold.value
      case "=":
        return value === threshold.value
      default:
        return false
    }
  }

  // Get the latest value for the selected metric
  const latestValue = metricData.length > 0 ? metricData[metricData.length - 1].value : null

  // Check if any thresholds are violated
  const violatedThresholds = activeThresholds.filter(
    (threshold) => latestValue !== null && checkThresholdViolation(latestValue, threshold),
  )

  return (
    <Card className="shadow-md">
      <CardHeader>
        <CardTitle>Threshold Visualization</CardTitle>
        <CardDescription>View metrics with configured thresholds</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="drift" onValueChange={(value) => setActiveTab(value as "drift" | "performance")}>
          <TabsList className="grid w-full grid-cols-2 mb-4">
            <TabsTrigger value="drift">Drift Metrics</TabsTrigger>
            <TabsTrigger value="performance">Performance Metrics</TabsTrigger>
          </TabsList>

          {["drift", "performance"].map((tabValue) => (
            <TabsContent key={tabValue} value={tabValue} className="space-y-4">
              <div className="space-y-4">
                <div>
                  <Select value={selectedMetric} onValueChange={setSelectedMetric}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select a metric" />
                    </SelectTrigger>
                    <SelectContent>
                      {availableMetrics[tabValue as "drift" | "performance"].map((metric) => (
                        <SelectItem key={metric.id} value={metric.id}>
                          {metric.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {metricData.length > 0 ? (
                  <div className="space-y-4">
                    <div className="h-[300px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={metricData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="date" />
                          <YAxis />
                          <Tooltip />
                          <Line
                            type="monotone"
                            dataKey="value"
                            stroke="#8884d8"
                            strokeWidth={2}
                            dot={{ r: 3 }}
                            activeDot={{ r: 5 }}
                          />
                          {activeThresholds.map((threshold) => (
                            <ReferenceLine
                              key={threshold.id}
                              y={threshold.value}
                              stroke={getThresholdColor(threshold.severity)}
                              strokeDasharray="3 3"
                              label={{
                                value: `${threshold.name} (${getOperatorSymbol(threshold.operator)}${threshold.value})`,
                                fill: getThresholdColor(threshold.severity),
                                fontSize: 12,
                              }}
                            />
                          ))}
                        </LineChart>
                      </ResponsiveContainer>
                    </div>

                    <div className="space-y-2">
                      <h3 className="text-sm font-medium">Active Thresholds</h3>
                      {activeThresholds.length > 0 ? (
                        <div className="space-y-2">
                          {activeThresholds.map((threshold) => (
                            <div
                              key={threshold.id}
                              className={`p-3 rounded-md border ${
                                latestValue !== null && checkThresholdViolation(latestValue, threshold)
                                  ? "bg-red-50 border-red-200"
                                  : "bg-gray-50 border-gray-200"
                              }`}
                            >
                              <div className="flex items-center justify-between">
                                <div className="flex items-center space-x-2">
                                  <Badge
                                    className={`${
                                      latestValue !== null && checkThresholdViolation(latestValue, threshold)
                                        ? "bg-red-500"
                                        : "bg-gray-500"
                                    } text-white`}
                                  >
                                    {threshold.severity}
                                  </Badge>
                                  <span className="font-medium">{threshold.name}</span>
                                </div>
                                <span className="text-sm">
                                  {getOperatorSymbol(threshold.operator)} {threshold.value}
                                </span>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="flex items-center justify-center p-4 text-center border rounded-md bg-gray-50">
                          <div>
                            <Info className="mx-auto h-8 w-8 text-gray-400" />
                            <p className="mt-2 text-sm text-gray-500">No active thresholds for this metric</p>
                          </div>
                        </div>
                      )}
                    </div>

                    {violatedThresholds.length > 0 && (
                      <div className="p-4 border rounded-md bg-red-50 border-red-200">
                        <div className="flex items-start space-x-3">
                          <AlertTriangle className="h-5 w-5 text-red-500 mt-0.5" />
                          <div>
                            <h3 className="font-medium text-red-800">Threshold Violations Detected</h3>
                            <p className="mt-1 text-sm text-red-700">
                              Current value ({latestValue}) violates {violatedThresholds.length} threshold
                              {violatedThresholds.length > 1 ? "s" : ""}
                            </p>
                            <ul className="mt-2 space-y-1 text-sm text-red-700">
                              {violatedThresholds.map((threshold) => (
                                <li key={threshold.id}>
                                  • {threshold.name}: {getOperatorSymbol(threshold.operator)} {threshold.value}
                                </li>
                              ))}
                            </ul>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-[300px] border rounded-md bg-gray-50">
                    <div className="text-center">
                      <Info className="mx-auto h-10 w-10 text-gray-400" />
                      <p className="mt-2 text-gray-500">No data available for this metric</p>
                    </div>
                  </div>
                )}
              </div>
            </TabsContent>
          ))}
        </Tabs>
      </CardContent>
    </Card>
  )
}
