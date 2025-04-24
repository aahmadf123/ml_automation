"use client"

import { useState, useEffect } from "react"
import { useWebSocketContext } from "@/components/websocket-provider"
import { HolographicCard } from "@/components/ui/holographic-card"
import { DataFlowViz } from "@/components/data-flow-viz"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { AlertCircle, CheckCircle, Clock, TrendingUp, Activity, Database, AlertTriangle } from "lucide-react"

// Define types for our monitoring data
interface ModelMetrics {
  modelId: string
  modelName: string
  rmse: number
  mse: number
  mae: number
  r2: number
  lastUpdated: Date
  status: "healthy" | "warning" | "critical"
}

interface DataDriftAlert {
  id: string
  feature: string
  currentValue: number
  threshold: number
  severity: "info" | "warning" | "critical"
  timestamp: Date
  status: "active" | "resolved"
}

interface SystemTelemetry {
  cpuUsage: number
  memoryUsage: number
  gpuUsage?: number
  diskUsage: number
  networkLatency: number
  timestamp: Date
}

interface PipelineStatus {
  ingestion: "running" | "completed" | "failed" | "idle"
  preprocessing: "running" | "completed" | "failed" | "idle"
  training: "running" | "completed" | "failed" | "idle"
  evaluation: "running" | "completed" | "failed" | "idle"
  deployment: "running" | "completed" | "failed" | "idle"
  lastUpdated: Date
}

// Mock data for initial rendering
const initialModelMetrics: ModelMetrics[] = [
  {
    modelId: "model1",
    modelName: "Gradient Boosting",
    rmse: 0.26,
    mse: 0.068,
    mae: 0.22,
    r2: 0.88,
    lastUpdated: new Date(),
    status: "healthy"
  },
  {
    modelId: "model2",
    modelName: "Random Forest",
    rmse: 0.31,
    mse: 0.096,
    mae: 0.25,
    r2: 0.82,
    lastUpdated: new Date(),
    status: "warning"
  }
]

const initialDataDriftAlerts: DataDriftAlert[] = [
  {
    id: "drift1",
    feature: "property_value",
    currentValue: 3.2,
    threshold: 5.0,
    severity: "warning",
    timestamp: new Date(),
    status: "active"
  },
  {
    id: "drift2",
    feature: "claim_amount",
    currentValue: 7.8,
    threshold: 6.0,
    severity: "critical",
    timestamp: new Date(),
    status: "active"
  }
]

const initialSystemTelemetry: SystemTelemetry = {
  cpuUsage: 45,
  memoryUsage: 62,
  gpuUsage: 78,
  diskUsage: 35,
  networkLatency: 120,
  timestamp: new Date()
}

const initialPipelineStatus: PipelineStatus = {
  ingestion: "completed",
  preprocessing: "completed",
  training: "running",
  evaluation: "idle",
  deployment: "idle",
  lastUpdated: new Date()
}

// System nodes for the data flow visualization
const systemNodes = [
  { id: "data-source", name: "Data Source", position: [-2, 0, 0], color: "#3b82f6" },
  { id: "ingestion", name: "Ingestion", position: [-1, 0, 0], color: "#10b981" },
  { id: "preprocessing", name: "Preprocessing", position: [0, 0, 0], color: "#f59e0b" },
  { id: "training", name: "Training", position: [1, 0, 0], color: "#ef4444" },
  { id: "evaluation", name: "Evaluation", position: [2, 0, 0], color: "#8b5cf6" },
  { id: "deployment", name: "Deployment", position: [3, 0, 0], color: "#ec4899" }
]

export function PipelineMonitor() {
  const { lastMessage, messageHistory } = useWebSocketContext()
  const [modelMetrics, setModelMetrics] = useState<ModelMetrics[]>(initialModelMetrics)
  const [dataDriftAlerts, setDataDriftAlerts] = useState<DataDriftAlert[]>(initialDataDriftAlerts)
  const [systemTelemetry, setSystemTelemetry] = useState<SystemTelemetry>(initialSystemTelemetry)
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatus>(initialPipelineStatus)
  const [activeTab, setActiveTab] = useState("overview")

  // Process incoming WebSocket messages
  useEffect(() => {
    if (!lastMessage) return

    // Handle model metrics updates
    if (lastMessage.type === "model_metrics") {
      const updatedMetrics = [...modelMetrics]
      const existingIndex = updatedMetrics.findIndex(m => m.modelId === lastMessage.data.modelId)
      
      if (existingIndex >= 0) {
        updatedMetrics[existingIndex] = {
          ...updatedMetrics[existingIndex],
          ...lastMessage.data,
          lastUpdated: new Date()
        }
      } else {
        updatedMetrics.push({
          ...lastMessage.data,
          lastUpdated: new Date()
        })
      }
      
      setModelMetrics(updatedMetrics)
    }
    
    // Handle data drift alerts
    if (lastMessage.type === "drift_alert") {
      const newAlert: DataDriftAlert = {
        id: Date.now().toString(),
        feature: lastMessage.data.feature,
        currentValue: lastMessage.data.current,
        threshold: lastMessage.data.threshold,
        severity: lastMessage.data.status === "warning" ? "warning" : "critical",
        timestamp: new Date(),
        status: "active"
      }
      
      setDataDriftAlerts(prev => [newAlert, ...prev])
    }
    
    // Handle system telemetry updates
    if (lastMessage.type === "system_telemetry") {
      setSystemTelemetry({
        ...lastMessage.data,
        timestamp: new Date()
      })
    }
    
    // Handle pipeline status updates
    if (lastMessage.type === "pipeline_status") {
      setPipelineStatus({
        ...lastMessage.data,
        lastUpdated: new Date()
      })
    }
  }, [lastMessage])

  // Helper function to get status badge color
  const getStatusColor = (status: string) => {
    switch (status) {
      case "healthy":
        return "bg-emerald-500/15 text-emerald-600"
      case "warning":
        return "bg-amber-500/15 text-amber-600"
      case "critical":
        return "bg-rose-500/15 text-rose-600"
      case "running":
        return "bg-sky-500/15 text-sky-600"
      case "completed":
        return "bg-emerald-500/15 text-emerald-600"
      case "failed":
        return "bg-rose-500/15 text-rose-600"
      case "idle":
        return "bg-gray-500/15 text-gray-600"
      default:
        return "bg-gray-500/15 text-gray-600"
    }
  }

  // Helper function to get status icon
  const getStatusIcon = (status: string) => {
    switch (status) {
      case "healthy":
      case "completed":
        return <CheckCircle className="h-4 w-4 text-emerald-600" />
      case "warning":
        return <AlertCircle className="h-4 w-4 text-amber-600" />
      case "critical":
      case "failed":
        return <AlertTriangle className="h-4 w-4 text-rose-600" />
      case "running":
        return <Activity className="h-4 w-4 text-sky-600" />
      case "idle":
        return <Clock className="h-4 w-4 text-gray-600" />
      default:
        return <Clock className="h-4 w-4 text-gray-600" />
    }
  }

  return (
    <div className="space-y-6 bg-gradient-to-b from-gray-50 to-gray-100 p-6 rounded-lg">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-800">Loss History Pipeline Monitor</h2>
        <Badge variant="outline" className="px-3 py-1 border-gray-300 text-gray-700">
          Real-time monitoring
        </Badge>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid grid-cols-4 mb-4 bg-white border border-gray-200">
          <TabsTrigger value="overview" className="text-gray-700 data-[state=active]:text-gray-900">Overview</TabsTrigger>
          <TabsTrigger value="models" className="text-gray-700 data-[state=active]:text-gray-900">Model Metrics</TabsTrigger>
          <TabsTrigger value="drift" className="text-gray-700 data-[state=active]:text-gray-900">Data Drift</TabsTrigger>
          <TabsTrigger value="system" className="text-gray-700 data-[state=active]:text-gray-900">System Telemetry</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card className="bg-gradient-to-br from-sky-500/5 via-indigo-500/5 to-violet-500/5">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Pipeline Status</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {Object.values(pipelineStatus).filter(status => status === "running").length}
                  <span className="text-sm font-normal text-muted-foreground"> active</span>
                </div>
                <div className="mt-2">
                  <Badge className={pipelineStatus.training === "running" ? "bg-blue-500/15 text-blue-600" : "bg-gray-500/15 text-gray-600"}>
                    {pipelineStatus.training === "running" ? "Training" : "Idle"}
                  </Badge>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-gradient-to-br from-amber-500/5 via-orange-500/5 to-rose-500/5">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Active Alerts</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {dataDriftAlerts.filter(alert => alert.status === "active").length}
                  <span className="text-sm font-normal text-muted-foreground"> alerts</span>
                </div>
                <div className="mt-2">
                  <Badge className={dataDriftAlerts.some(alert => alert.severity === "critical") ? "bg-red-500/15 text-red-600" : "bg-amber-500/15 text-amber-600"}>
                    {dataDriftAlerts.some(alert => alert.severity === "critical") ? "Critical" : "Warning"}
                  </Badge>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-gradient-to-br from-emerald-500/5 via-teal-500/5 to-cyan-500/5">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Model Performance</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {modelMetrics.filter(m => m.status === "healthy").length}
                  <span className="text-sm font-normal text-muted-foreground"> / {modelMetrics.length}</span>
                </div>
                <div className="mt-2">
                  <Badge className={modelMetrics.some(m => m.status === "critical") ? "bg-red-500/15 text-red-600" : "bg-emerald-500/15 text-emerald-600"}>
                    {modelMetrics.some(m => m.status === "critical") ? "Critical" : "Healthy"}
                  </Badge>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-gradient-to-br from-violet-500/5 via-purple-500/5 to-fuchsia-500/5">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">System Health</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {systemTelemetry.cpuUsage}
                  <span className="text-sm font-normal text-muted-foreground">% CPU</span>
                </div>
                <div className="mt-2">
                  <Badge className={systemTelemetry.cpuUsage > 80 ? "bg-red-500/15 text-red-600" : "bg-emerald-500/15 text-emerald-600"}>
                    {systemTelemetry.cpuUsage > 80 ? "High Load" : "Normal"}
                  </Badge>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card className="bg-white shadow-sm border border-gray-200">
            <CardHeader>
              <CardTitle className="text-gray-800">Pipeline Data Flow</CardTitle>
              <CardDescription className="text-gray-500">Real-time visualization of data movement through the pipeline</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <DataFlowViz nodes={systemNodes} />
              </div>
            </CardContent>
          </Card>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card className="bg-white shadow-sm border border-gray-200">
              <CardHeader>
                <CardTitle className="text-gray-800">Recent Alerts</CardTitle>
                <CardDescription className="text-gray-500">Latest data drift and system alerts</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                {dataDriftAlerts.slice(0, 3).map(alert => (
                  <Alert key={alert.id} variant={alert.severity === "critical" ? "destructive" : "default"} className="border border-gray-200">
                    <AlertCircle className="h-4 w-4 text-amber-600" />
                    <AlertTitle className="text-gray-800">{alert.feature} Drift Detected</AlertTitle>
                    <AlertDescription className="text-gray-600">
                      Current value: {alert.currentValue.toFixed(2)} (threshold: {alert.threshold.toFixed(2)})
                    </AlertDescription>
                  </Alert>
                ))}
              </CardContent>
            </Card>

            <Card className="bg-white shadow-sm border border-gray-200">
              <CardHeader>
                <CardTitle className="text-gray-800">Pipeline Stages</CardTitle>
                <CardDescription className="text-gray-500">Current status of each pipeline stage</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                {Object.entries(pipelineStatus)
                  .filter(([key]) => key !== "lastUpdated")
                  .map(([stage, status]) => (
                    <div key={stage} className="flex items-center justify-between py-2 border-b border-gray-200 last:border-0">
                      <div className="flex items-center space-x-2">
                        <span className="capitalize text-gray-700">{stage}</span>
                        <Badge className={getStatusColor(status)}>
                          {getStatusIcon(status)}
                          <span className="ml-1">{status}</span>
                        </Badge>
                      </div>
                    </div>
                  ))}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="models" className="space-y-4">
          <Card className="bg-white shadow-sm border border-gray-200">
            <CardHeader>
              <CardTitle className="text-gray-800">Model Performance Metrics</CardTitle>
              <CardDescription className="text-gray-500">Real-time metrics for all deployed models</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-gray-200">
                      <th className="text-left py-2 text-gray-700">Model</th>
                      <th className="text-right py-2 text-gray-700">RMSE</th>
                      <th className="text-right py-2 text-gray-700">MSE</th>
                      <th className="text-right py-2 text-gray-700">MAE</th>
                      <th className="text-right py-2 text-gray-700">R²</th>
                      <th className="text-right py-2 text-gray-700">Status</th>
                      <th className="text-right py-2 text-gray-700">Last Updated</th>
                    </tr>
                  </thead>
                  <tbody>
                    {modelMetrics.map(model => (
                      <tr key={model.modelId} className="border-b border-gray-200">
                        <td className="py-2 text-gray-800">{model.modelName}</td>
                        <td className="text-right text-gray-800">{model.rmse.toFixed(4)}</td>
                        <td className="text-right text-gray-800">{model.mse.toFixed(4)}</td>
                        <td className="text-right text-gray-800">{model.mae.toFixed(4)}</td>
                        <td className="text-right text-gray-800">{model.r2.toFixed(4)}</td>
                        <td className="text-right">
                          <Badge className={getStatusColor(model.status)}>
                            {getStatusIcon(model.status)}
                            <span className="ml-1">{model.status}</span>
                          </Badge>
                        </td>
                        <td className="text-right text-sm text-gray-500">
                          {model.lastUpdated.toLocaleTimeString()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="drift" className="space-y-4">
          <Card className="bg-white shadow-sm border border-gray-200">
            <CardHeader>
              <CardTitle className="text-gray-800">Data Drift Alerts</CardTitle>
              <CardDescription className="text-gray-500">Features with detected distribution shifts</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-gray-200">
                      <th className="text-left py-2 text-gray-700">Feature</th>
                      <th className="text-right py-2 text-gray-700">Current Value</th>
                      <th className="text-right py-2 text-gray-700">Threshold</th>
                      <th className="text-right py-2 text-gray-700">Severity</th>
                      <th className="text-right py-2 text-gray-700">Status</th>
                      <th className="text-right py-2 text-gray-700">Detected At</th>
                    </tr>
                  </thead>
                  <tbody>
                    {dataDriftAlerts.map(alert => (
                      <tr key={alert.id} className="border-b border-gray-200">
                        <td className="py-2 text-gray-800">{alert.feature}</td>
                        <td className="text-right text-gray-800">{alert.currentValue.toFixed(4)}</td>
                        <td className="text-right text-gray-800">{alert.threshold.toFixed(4)}</td>
                        <td className="text-right">
                          <Badge className={getStatusColor(alert.severity)}>
                            {getStatusIcon(alert.severity)}
                            <span className="ml-1">{alert.severity}</span>
                          </Badge>
                        </td>
                        <td className="text-right">
                          <Badge className={alert.status === "active" ? "bg-sky-500/15 text-sky-600" : "bg-gray-500/15 text-gray-600"}>
                            {alert.status}
                          </Badge>
                        </td>
                        <td className="text-right text-sm text-gray-500">
                          {alert.timestamp.toLocaleTimeString()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="system" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <Card className="bg-gradient-to-br from-sky-500/5 via-indigo-500/5 to-violet-500/5">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">CPU Usage</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {systemTelemetry.cpuUsage}
                  <span className="text-sm font-normal text-muted-foreground">%</span>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-gradient-to-br from-amber-500/5 via-orange-500/5 to-rose-500/5">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Memory Usage</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {systemTelemetry.memoryUsage}
                  <span className="text-sm font-normal text-muted-foreground">%</span>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-gradient-to-br from-emerald-500/5 via-teal-500/5 to-cyan-500/5">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">GPU Usage</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {systemTelemetry.gpuUsage || 0}
                  <span className="text-sm font-normal text-muted-foreground">%</span>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-gradient-to-br from-violet-500/5 via-purple-500/5 to-fuchsia-500/5">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Disk Usage</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {systemTelemetry.diskUsage}
                  <span className="text-sm font-normal text-muted-foreground">%</span>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-gradient-to-br from-pink-500/5 via-rose-500/5 to-red-500/5">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Network Latency</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {systemTelemetry.networkLatency}
                  <span className="text-sm font-normal text-muted-foreground"> ms</span>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card className="bg-white shadow-sm border border-gray-200">
            <CardHeader>
              <CardTitle className="text-gray-800">System Health</CardTitle>
              <CardDescription className="text-gray-500">Last updated: {systemTelemetry.timestamp.toLocaleTimeString()}</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-gray-700">CPU Usage</span>
                    <span className="text-gray-800">{systemTelemetry.cpuUsage}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2.5">
                    <div 
                      className={`h-2.5 rounded-full ${
                        systemTelemetry.cpuUsage > 80 ? "bg-rose-500" : 
                        systemTelemetry.cpuUsage > 60 ? "bg-amber-500" : "bg-emerald-500"
                      }`} 
                      style={{ width: `${systemTelemetry.cpuUsage}%` }}
                    ></div>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-gray-700">Memory Usage</span>
                    <span className="text-gray-800">{systemTelemetry.memoryUsage}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2.5">
                    <div 
                      className={`h-2.5 rounded-full ${
                        systemTelemetry.memoryUsage > 80 ? "bg-rose-500" : 
                        systemTelemetry.memoryUsage > 60 ? "bg-amber-500" : "bg-emerald-500"
                      }`} 
                      style={{ width: `${systemTelemetry.memoryUsage}%` }}
                    ></div>
                  </div>
                </div>
                
                {systemTelemetry.gpuUsage !== undefined && (
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-700">GPU Usage</span>
                      <span className="text-gray-800">{systemTelemetry.gpuUsage}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2.5">
                      <div 
                        className={`h-2.5 rounded-full ${
                          systemTelemetry.gpuUsage > 80 ? "bg-rose-500" : 
                          systemTelemetry.gpuUsage > 60 ? "bg-amber-500" : "bg-emerald-500"
                        }`} 
                        style={{ width: `${systemTelemetry.gpuUsage}%` }}
                      ></div>
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
} 