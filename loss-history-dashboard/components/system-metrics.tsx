"use client"

import { useState, useEffect } from "react"
import { useWebSocketContext } from "@/components/websocket-provider"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { AlertCircle, CheckCircle, Clock, Activity, Database, Server, Zap } from "lucide-react"
import { LineChart } from "@/components/ui/line-chart"

// Define types for our metrics data
interface SystemMetric {
  name: string
  value: number
  unit: string
  timestamp: Date
  status: "healthy" | "warning" | "critical"
  threshold: number
}

interface DAGMetric {
  dagId: string
  runtime: number
  status: "success" | "running" | "failed" | "skipped"
  lastRun: Date
  avgRuntime: number
  successRate: number
}

interface MLflowMetric {
  endpoint: string
  latency: number
  status: "healthy" | "warning" | "critical"
  lastCheck: Date
  avgLatency: number
  errorRate: number
}

// Mock data for initial rendering
const initialSystemMetrics: SystemMetric[] = [
  {
    name: "CPU Usage",
    value: 45,
    unit: "%",
    timestamp: new Date(),
    status: "healthy",
    threshold: 80
  },
  {
    name: "Memory Usage",
    value: 62,
    unit: "%",
    timestamp: new Date(),
    status: "warning",
    threshold: 70
  },
  {
    name: "Disk Usage",
    value: 78,
    unit: "%",
    timestamp: new Date(),
    status: "warning",
    threshold: 80
  },
  {
    name: "Network I/O",
    value: 120,
    unit: "MB/s",
    timestamp: new Date(),
    status: "healthy",
    threshold: 200
  }
]

const initialDAGMetrics: DAGMetric[] = [
  {
    dagId: "data_ingestion",
    runtime: 120,
    status: "success",
    lastRun: new Date(Date.now() - 3600000),
    avgRuntime: 115,
    successRate: 98
  },
  {
    dagId: "feature_engineering",
    runtime: 180,
    status: "running",
    lastRun: new Date(Date.now() - 1800000),
    avgRuntime: 175,
    successRate: 95
  },
  {
    dagId: "model_training",
    runtime: 600,
    status: "success",
    lastRun: new Date(Date.now() - 7200000),
    avgRuntime: 580,
    successRate: 92
  },
  {
    dagId: "model_evaluation",
    runtime: 90,
    status: "failed",
    lastRun: new Date(Date.now() - 900000),
    avgRuntime: 85,
    successRate: 90
  }
]

const initialMLflowMetrics: MLflowMetric[] = [
  {
    endpoint: "/api/2.0/mlflow/runs/search",
    latency: 120,
    status: "healthy",
    lastCheck: new Date(Date.now() - 300000),
    avgLatency: 110,
    errorRate: 0.5
  },
  {
    endpoint: "/api/2.0/mlflow/metrics/get-history",
    latency: 180,
    status: "warning",
    lastCheck: new Date(Date.now() - 300000),
    avgLatency: 150,
    errorRate: 1.2
  },
  {
    endpoint: "/api/2.0/mlflow/artifacts/list",
    latency: 250,
    status: "critical",
    lastCheck: new Date(Date.now() - 300000),
    avgLatency: 200,
    errorRate: 3.5
  }
]

export function SystemMetrics() {
  const { lastMessage } = useWebSocketContext()
  const [systemMetrics, setSystemMetrics] = useState<SystemMetric[]>(initialSystemMetrics)
  const [dagMetrics, setDAGMetrics] = useState<DAGMetric[]>(initialDAGMetrics)
  const [mlflowMetrics, setMLflowMetrics] = useState<MLflowMetric[]>(initialMLflowMetrics)
  const [activeTab, setActiveTab] = useState("system")
  const [timeRange, setTimeRange] = useState("1h")

  // Process incoming WebSocket messages
  useEffect(() => {
    if (!lastMessage) return

    // Handle system metrics updates
    if (lastMessage.type === "system_metrics") {
      setSystemMetrics(prev => 
        prev.map(metric => 
          metric.name === lastMessage.data.name 
            ? { ...metric, ...lastMessage.data, timestamp: new Date() } 
            : metric
        )
      )
    }
    
    // Handle DAG metrics updates
    if (lastMessage.type === "dag_metrics") {
      setDAGMetrics(prev => 
        prev.map(metric => 
          metric.dagId === lastMessage.data.dagId 
            ? { ...metric, ...lastMessage.data, lastRun: new Date() } 
            : metric
        )
      )
    }
    
    // Handle MLflow metrics updates
    if (lastMessage.type === "mlflow_metrics") {
      setMLflowMetrics(prev => 
        prev.map(metric => 
          metric.endpoint === lastMessage.data.endpoint 
            ? { ...metric, ...lastMessage.data, lastCheck: new Date() } 
            : metric
        )
      )
    }
  }, [lastMessage])

  // Helper function to get status badge color
  const getStatusColor = (status: string) => {
    switch (status) {
      case "healthy":
      case "success":
        return "bg-emerald-500/15 text-emerald-600"
      case "warning":
        return "bg-amber-500/15 text-amber-600"
      case "critical":
      case "failed":
        return "bg-rose-500/15 text-rose-600"
      case "running":
        return "bg-sky-500/15 text-sky-600"
      case "skipped":
        return "bg-gray-500/15 text-gray-600"
      default:
        return "bg-gray-500/15 text-gray-600"
    }
  }

  // Helper function to get status icon
  const getStatusIcon = (status: string) => {
    switch (status) {
      case "healthy":
      case "success":
        return <CheckCircle className="h-4 w-4 text-emerald-600" />
      case "warning":
        return <AlertCircle className="h-4 w-4 text-amber-600" />
      case "critical":
      case "failed":
        return <AlertCircle className="h-4 w-4 text-rose-600" />
      case "running":
        return <Activity className="h-4 w-4 text-sky-600" />
      case "skipped":
        return <Clock className="h-4 w-4 text-gray-600" />
      default:
        return <Clock className="h-4 w-4 text-gray-600" />
    }
  }

  // Generate mock time series data for charts
  const generateTimeSeriesData = (baseValue: number, variance: number, points: number = 20) => {
    return Array.from({ length: points }, (_, i) => ({
      timestamp: new Date(Date.now() - (points - i) * 300000).toISOString(),
      value: baseValue + (Math.random() - 0.5) * variance
    }))
  }

  return (
    <div className="space-y-6 bg-gradient-to-b from-gray-50 to-gray-100 p-6 rounded-lg">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-800">System Metrics</h2>
        <div className="flex items-center space-x-2">
          <select 
            value={timeRange} 
            onChange={(e) => setTimeRange(e.target.value)}
            className="bg-white border border-gray-200 rounded-md px-3 py-1.5 text-sm"
          >
            <option value="1h">Last Hour</option>
            <option value="6h">Last 6 Hours</option>
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
          </select>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid grid-cols-3 mb-4 bg-white border border-gray-200">
          <TabsTrigger value="system" className="text-gray-700 data-[state=active]:text-gray-900">System Resources</TabsTrigger>
          <TabsTrigger value="dags" className="text-gray-700 data-[state=active]:text-gray-900">DAG Performance</TabsTrigger>
          <TabsTrigger value="mlflow" className="text-gray-700 data-[state=active]:text-gray-900">MLflow API</TabsTrigger>
        </TabsList>

        <TabsContent value="system" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {systemMetrics.map((metric, index) => (
              <Card key={index} className="bg-white shadow-sm border border-gray-200">
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-sm font-medium text-gray-700">{metric.name}</CardTitle>
                    <Badge className={getStatusColor(metric.status)}>
                      {getStatusIcon(metric.status)}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-gray-800">{metric.value}{metric.unit}</div>
                  <div className="mt-2 h-20">
                    <LineChart 
                      data={generateTimeSeriesData(metric.value, 10)} 
                      color={metric.status === "healthy" ? "rgb(16, 185, 129)" : 
                             metric.status === "warning" ? "rgb(245, 158, 11)" : "rgb(239, 68, 68)"}
                    />
                  </div>
                  <div className="mt-2 text-xs text-gray-500">
                    Threshold: {metric.threshold}{metric.unit}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          <Card className="bg-white shadow-sm border border-gray-200">
            <CardHeader>
              <CardTitle className="text-gray-800">Resource Utilization Trends</CardTitle>
              <CardDescription className="text-gray-500">Historical view of system resource usage</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <LineChart 
                  data={generateTimeSeriesData(60, 20, 40)} 
                  color="rgb(99, 102, 241)"
                  showLegend={true}
                  legendItems={[
                    { label: "CPU", color: "rgb(16, 185, 129)" },
                    { label: "Memory", color: "rgb(245, 158, 11)" },
                    { label: "Disk", color: "rgb(239, 68, 68)" }
                  ]}
                />
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="dags" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card className="bg-white shadow-sm border border-gray-200">
              <CardHeader>
                <CardTitle className="text-gray-800">DAG Runtime Summary</CardTitle>
                <CardDescription className="text-gray-500">Performance metrics for each DAG</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-gray-200">
                        <th className="text-left py-2 text-gray-700">DAG</th>
                        <th className="text-right py-2 text-gray-700">Runtime</th>
                        <th className="text-right py-2 text-gray-700">Avg Runtime</th>
                        <th className="text-right py-2 text-gray-700">Success Rate</th>
                        <th className="text-right py-2 text-gray-700">Status</th>
                        <th className="text-right py-2 text-gray-700">Last Run</th>
                      </tr>
                    </thead>
                    <tbody>
                      {dagMetrics.map((dag, index) => (
                        <tr key={index} className="border-b border-gray-200">
                          <td className="py-2 text-gray-800">{dag.dagId}</td>
                          <td className="text-right text-gray-800">{dag.runtime}s</td>
                          <td className="text-right text-gray-800">{dag.avgRuntime}s</td>
                          <td className="text-right text-gray-800">{dag.successRate}%</td>
                          <td className="text-right">
                            <Badge className={getStatusColor(dag.status)}>
                              {getStatusIcon(dag.status)}
                              <span className="ml-1">{dag.status}</span>
                            </Badge>
                          </td>
                          <td className="text-right text-sm text-gray-500">
                            {dag.lastRun.toLocaleTimeString()}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-white shadow-sm border border-gray-200">
              <CardHeader>
                <CardTitle className="text-gray-800">DAG Runtime Trends</CardTitle>
                <CardDescription className="text-gray-500">Historical view of DAG execution times</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <LineChart 
                    data={generateTimeSeriesData(180, 30, 40)} 
                    color="rgb(99, 102, 241)"
                    showLegend={true}
                    legendItems={[
                      { label: "Data Ingestion", color: "rgb(16, 185, 129)" },
                      { label: "Feature Engineering", color: "rgb(245, 158, 11)" },
                      { label: "Model Training", color: "rgb(239, 68, 68)" },
                      { label: "Model Evaluation", color: "rgb(99, 102, 241)" }
                    ]}
                  />
                </div>
              </CardContent>
            </Card>
          </div>

          <Card className="bg-white shadow-sm border border-gray-200">
            <CardHeader>
              <CardTitle className="text-gray-800">DAG Failures</CardTitle>
              <CardDescription className="text-gray-500">Recent DAG execution failures</CardDescription>
            </CardHeader>
            <CardContent>
              {dagMetrics.filter(dag => dag.status === "failed").length > 0 ? (
                <div className="space-y-4">
                  {dagMetrics.filter(dag => dag.status === "failed").map((dag, index) => (
                    <Alert key={index} variant="destructive" className="border border-gray-200">
                      <AlertCircle className="h-4 w-4 text-rose-600" />
                      <AlertTitle className="text-gray-800">{dag.dagId} Failed</AlertTitle>
                      <AlertDescription className="text-gray-600">
                        Last run at {dag.lastRun.toLocaleString()} with runtime of {dag.runtime}s
                      </AlertDescription>
                    </Alert>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8">
                  <CheckCircle className="h-12 w-12 mx-auto text-emerald-500" />
                  <p className="mt-2 text-sm text-gray-500">No recent DAG failures</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="mlflow" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card className="bg-white shadow-sm border border-gray-200">
              <CardHeader>
                <CardTitle className="text-gray-800">MLflow API Latency</CardTitle>
                <CardDescription className="text-gray-500">Response times for MLflow API endpoints</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-gray-200">
                        <th className="text-left py-2 text-gray-700">Endpoint</th>
                        <th className="text-right py-2 text-gray-700">Latency</th>
                        <th className="text-right py-2 text-gray-700">Avg Latency</th>
                        <th className="text-right py-2 text-gray-700">Error Rate</th>
                        <th className="text-right py-2 text-gray-700">Status</th>
                        <th className="text-right py-2 text-gray-700">Last Check</th>
                      </tr>
                    </thead>
                    <tbody>
                      {mlflowMetrics.map((metric, index) => (
                        <tr key={index} className="border-b border-gray-200">
                          <td className="py-2 text-gray-800">{metric.endpoint}</td>
                          <td className="text-right text-gray-800">{metric.latency}ms</td>
                          <td className="text-right text-gray-800">{metric.avgLatency}ms</td>
                          <td className="text-right text-gray-800">{metric.errorRate}%</td>
                          <td className="text-right">
                            <Badge className={getStatusColor(metric.status)}>
                              {getStatusIcon(metric.status)}
                              <span className="ml-1">{metric.status}</span>
                            </Badge>
                          </td>
                          <td className="text-right text-sm text-gray-500">
                            {metric.lastCheck.toLocaleTimeString()}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-white shadow-sm border border-gray-200">
              <CardHeader>
                <CardTitle className="text-gray-800">MLflow API Trends</CardTitle>
                <CardDescription className="text-gray-500">Historical view of API response times</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <LineChart 
                    data={generateTimeSeriesData(150, 50, 40)} 
                    color="rgb(99, 102, 241)"
                    showLegend={true}
                    legendItems={[
                      { label: "/runs/search", color: "rgb(16, 185, 129)" },
                      { label: "/metrics/get-history", color: "rgb(245, 158, 11)" },
                      { label: "/artifacts/list", color: "rgb(239, 68, 68)" }
                    ]}
                  />
                </div>
              </CardContent>
            </Card>
          </div>

          <Card className="bg-white shadow-sm border border-gray-200">
            <CardHeader>
              <CardTitle className="text-gray-800">MLflow API Issues</CardTitle>
              <CardDescription className="text-gray-500">Recent API performance issues</CardDescription>
            </CardHeader>
            <CardContent>
              {mlflowMetrics.filter(metric => metric.status !== "healthy").length > 0 ? (
                <div className="space-y-4">
                  {mlflowMetrics.filter(metric => metric.status !== "healthy").map((metric, index) => (
                    <Alert 
                      key={index} 
                      variant={metric.status === "critical" ? "destructive" : "default"} 
                      className="border border-gray-200"
                    >
                      <AlertCircle className={`h-4 w-4 ${
                        metric.status === "critical" ? "text-rose-600" : "text-amber-600"
                      }`} />
                      <AlertTitle className="text-gray-800">{metric.endpoint} {metric.status}</AlertTitle>
                      <AlertDescription className="text-gray-600">
                        Current latency: {metric.latency}ms (avg: {metric.avgLatency}ms), Error rate: {metric.errorRate}%
                      </AlertDescription>
                    </Alert>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8">
                  <CheckCircle className="h-12 w-12 mx-auto text-emerald-500" />
                  <p className="mt-2 text-sm text-gray-500">No recent API issues</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
} 