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
        return "bg-green-500/20 text-green-500"
      case "warning":
        return "bg-yellow-500/20 text-yellow-500"
      case "critical":
        return "bg-red-500/20 text-red-500"
      case "running":
        return "bg-blue-500/20 text-blue-500"
      case "completed":
        return "bg-green-500/20 text-green-500"
      case "failed":
        return "bg-red-500/20 text-red-500"
      case "idle":
        return "bg-gray-500/20 text-gray-500"
      default:
        return "bg-gray-500/20 text-gray-500"
    }
  }

  // Helper function to get status icon
  const getStatusIcon = (status: string) => {
    switch (status) {
      case "healthy":
      case "completed":
        return <CheckCircle className="h-4 w-4" />
      case "warning":
        return <AlertCircle className="h-4 w-4" />
      case "critical":
      case "failed":
        return <AlertTriangle className="h-4 w-4" />
      case "running":
        return <Activity className="h-4 w-4" />
      case "idle":
        return <Clock className="h-4 w-4" />
      default:
        return <Clock className="h-4 w-4" />
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Loss History Pipeline Monitor</h2>
        <Badge variant="outline" className="px-3 py-1">
          Real-time monitoring
        </Badge>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid grid-cols-4 mb-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="models">Model Metrics</TabsTrigger>
          <TabsTrigger value="drift">Data Drift</TabsTrigger>
          <TabsTrigger value="system">System Telemetry</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <HolographicCard
              title="Pipeline Status"
              value={Object.values(pipelineStatus).filter(status => status === "running").length}
              unit=" active"
              gradient="from-blue-500/20 via-indigo-500/20 to-purple-500/20"
              glowColor="rgba(99, 102, 241, 0.5)"
              badge={{
                text: pipelineStatus.training === "running" ? "Training" : "Idle",
                color: pipelineStatus.training === "running" ? "#3b82f6" : "#9ca3af"
              }}
            />
            <HolographicCard
              title="Active Alerts"
              value={dataDriftAlerts.filter(alert => alert.status === "active").length}
              unit=" alerts"
              gradient="from-amber-500/20 via-orange-500/20 to-red-500/20"
              glowColor="rgba(245, 158, 11, 0.5)"
              badge={{
                text: dataDriftAlerts.some(alert => alert.severity === "critical") ? "Critical" : "Warning",
                color: dataDriftAlerts.some(alert => alert.severity === "critical") ? "#ef4444" : "#f59e0b"
              }}
            />
            <HolographicCard
              title="Model Performance"
              value={modelMetrics.filter(m => m.status === "healthy").length}
              unit=" / " + modelMetrics.length
              gradient="from-emerald-500/20 via-teal-500/20 to-cyan-500/20"
              glowColor="rgba(16, 185, 129, 0.5)"
              badge={{
                text: modelMetrics.some(m => m.status === "critical") ? "Critical" : "Healthy",
                color: modelMetrics.some(m => m.status === "critical") ? "#ef4444" : "#10b981"
              }}
            />
            <HolographicCard
              title="System Health"
              value={systemTelemetry.cpuUsage}
              unit="% CPU"
              gradient="from-violet-500/20 via-purple-500/20 to-fuchsia-500/20"
              glowColor="rgba(139, 92, 246, 0.5)"
              badge={{
                text: systemTelemetry.cpuUsage > 80 ? "High Load" : "Normal",
                color: systemTelemetry.cpuUsage > 80 ? "#ef4444" : "#10b981"
              }}
            />
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Pipeline Data Flow</CardTitle>
              <CardDescription>Real-time visualization of data movement through the pipeline</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <DataFlowViz nodes={systemNodes} />
              </div>
            </CardContent>
          </Card>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Recent Alerts</CardTitle>
                <CardDescription>Latest data drift and system alerts</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                {dataDriftAlerts.slice(0, 3).map(alert => (
                  <Alert key={alert.id} variant={alert.severity === "critical" ? "destructive" : "default"}>
                    <AlertCircle className="h-4 w-4" />
                    <AlertTitle>{alert.feature} Drift Detected</AlertTitle>
                    <AlertDescription>
                      Current value: {alert.currentValue.toFixed(2)} (threshold: {alert.threshold.toFixed(2)})
                    </AlertDescription>
                  </Alert>
                ))}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Pipeline Stages</CardTitle>
                <CardDescription>Current status of each pipeline stage</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                {Object.entries(pipelineStatus)
                  .filter(([key]) => key !== "lastUpdated")
                  .map(([stage, status]) => (
                    <div key={stage} className="flex items-center justify-between py-2 border-b last:border-0">
                      <div className="flex items-center space-x-2">
                        <span className="capitalize">{stage}</span>
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
          <Card>
            <CardHeader>
              <CardTitle>Model Performance Metrics</CardTitle>
              <CardDescription>Real-time metrics for all deployed models</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-2">Model</th>
                      <th className="text-right py-2">RMSE</th>
                      <th className="text-right py-2">MSE</th>
                      <th className="text-right py-2">MAE</th>
                      <th className="text-right py-2">R²</th>
                      <th className="text-right py-2">Status</th>
                      <th className="text-right py-2">Last Updated</th>
                    </tr>
                  </thead>
                  <tbody>
                    {modelMetrics.map(model => (
                      <tr key={model.modelId} className="border-b">
                        <td className="py-2">{model.modelName}</td>
                        <td className="text-right">{model.rmse.toFixed(4)}</td>
                        <td className="text-right">{model.mse.toFixed(4)}</td>
                        <td className="text-right">{model.mae.toFixed(4)}</td>
                        <td className="text-right">{model.r2.toFixed(4)}</td>
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
          <Card>
            <CardHeader>
              <CardTitle>Data Drift Alerts</CardTitle>
              <CardDescription>Features with detected distribution shifts</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-2">Feature</th>
                      <th className="text-right py-2">Current Value</th>
                      <th className="text-right py-2">Threshold</th>
                      <th className="text-right py-2">Severity</th>
                      <th className="text-right py-2">Status</th>
                      <th className="text-right py-2">Detected At</th>
                    </tr>
                  </thead>
                  <tbody>
                    {dataDriftAlerts.map(alert => (
                      <tr key={alert.id} className="border-b">
                        <td className="py-2">{alert.feature}</td>
                        <td className="text-right">{alert.currentValue.toFixed(4)}</td>
                        <td className="text-right">{alert.threshold.toFixed(4)}</td>
                        <td className="text-right">
                          <Badge className={getStatusColor(alert.severity)}>
                            {getStatusIcon(alert.severity)}
                            <span className="ml-1">{alert.severity}</span>
                          </Badge>
                        </td>
                        <td className="text-right">
                          <Badge className={alert.status === "active" ? "bg-blue-500/20 text-blue-500" : "bg-gray-500/20 text-gray-500"}>
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
            <HolographicCard
              title="CPU Usage"
              value={systemTelemetry.cpuUsage}
              unit="%"
              gradient="from-blue-500/20 via-indigo-500/20 to-purple-500/20"
              glowColor="rgba(99, 102, 241, 0.5)"
            />
            <HolographicCard
              title="Memory Usage"
              value={systemTelemetry.memoryUsage}
              unit="%"
              gradient="from-amber-500/20 via-orange-500/20 to-red-500/20"
              glowColor="rgba(245, 158, 11, 0.5)"
            />
            <HolographicCard
              title="GPU Usage"
              value={systemTelemetry.gpuUsage || 0}
              unit="%"
              gradient="from-emerald-500/20 via-teal-500/20 to-cyan-500/20"
              glowColor="rgba(16, 185, 129, 0.5)"
            />
            <HolographicCard
              title="Disk Usage"
              value={systemTelemetry.diskUsage}
              unit="%"
              gradient="from-violet-500/20 via-purple-500/20 to-fuchsia-500/20"
              glowColor="rgba(139, 92, 246, 0.5)"
            />
            <HolographicCard
              title="Network Latency"
              value={systemTelemetry.networkLatency}
              unit=" ms"
              gradient="from-pink-500/20 via-rose-500/20 to-red-500/20"
              glowColor="rgba(236, 72, 153, 0.5)"
            />
          </div>

          <Card>
            <CardHeader>
              <CardTitle>System Health</CardTitle>
              <CardDescription>Last updated: {systemTelemetry.timestamp.toLocaleTimeString()}</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>CPU Usage</span>
                    <span>{systemTelemetry.cpuUsage}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2.5">
                    <div 
                      className={`h-2.5 rounded-full ${
                        systemTelemetry.cpuUsage > 80 ? "bg-red-500" : 
                        systemTelemetry.cpuUsage > 60 ? "bg-yellow-500" : "bg-green-500"
                      }`} 
                      style={{ width: `${systemTelemetry.cpuUsage}%` }}
                    ></div>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>Memory Usage</span>
                    <span>{systemTelemetry.memoryUsage}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2.5">
                    <div 
                      className={`h-2.5 rounded-full ${
                        systemTelemetry.memoryUsage > 80 ? "bg-red-500" : 
                        systemTelemetry.memoryUsage > 60 ? "bg-yellow-500" : "bg-green-500"
                      }`} 
                      style={{ width: `${systemTelemetry.memoryUsage}%` }}
                    ></div>
                  </div>
                </div>
                
                {systemTelemetry.gpuUsage !== undefined && (
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span>GPU Usage</span>
                      <span>{systemTelemetry.gpuUsage}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2.5">
                      <div 
                        className={`h-2.5 rounded-full ${
                          systemTelemetry.gpuUsage > 80 ? "bg-red-500" : 
                          systemTelemetry.gpuUsage > 60 ? "bg-yellow-500" : "bg-green-500"
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