"use client"

import { useEffect, useState } from "react"
import { Card } from "./ui/card"
import { LineChart } from "./ui/line-chart"

interface Metric {
  timestamp: number
  value: number
}

interface SystemMetrics {
  runtime: number
  memory_usage: number
}

interface ModelMetrics {
  [modelName: string]: {
    loss: Metric[]
    accuracy: Metric[]
  }
}

export function RealTimeMetrics() {
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics>({
    runtime: 0,
    memory_usage: 0
  })
  const [modelMetrics, setModelMetrics] = useState<ModelMetrics>({})
  const [ws, setWs] = useState<WebSocket | null>(null)

  useEffect(() => {
    const websocket = new WebSocket("ws://localhost:8000")

    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data)
      
      if (data.type === "system_metrics") {
        setSystemMetrics({
          runtime: data.runtime,
          memory_usage: data.memory_usage
        })
      } else if (data.type === "model_metrics") {
        setModelMetrics((prev) => ({
          ...prev,
          [data.model_name]: {
            loss: [...(prev[data.model_name]?.loss || []), { timestamp: Date.now(), value: data.loss }],
            accuracy: [...(prev[data.model_name]?.accuracy || []), { timestamp: Date.now(), value: data.accuracy }]
          }
        }))
      }
    }

    websocket.onerror = (error) => {
      console.error("WebSocket error:", error)
    }

    setWs(websocket)

    return () => {
      websocket.close()
    }
  }, [])

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      <Card title="System Metrics">
        <div className="space-y-4">
          <div>
            <h4 className="text-sm font-medium">Runtime (seconds)</h4>
            <p className="text-2xl font-bold">{systemMetrics.runtime.toFixed(2)}</p>
          </div>
          <div>
            <h4 className="text-sm font-medium">Memory Usage (MB)</h4>
            <p className="text-2xl font-bold">{systemMetrics.memory_usage.toFixed(2)}</p>
          </div>
        </div>
      </Card>

      {Object.entries(modelMetrics).map(([modelName, metrics]) => (
        <div key={modelName} className="space-y-4">
          <Card title={`${modelName} Loss`}>
            <LineChart data={metrics.loss} color="#ef4444" />
          </Card>
          <Card title={`${modelName} Accuracy`}>
            <LineChart data={metrics.accuracy} color="#22c55e" />
          </Card>
        </div>
      ))}
    </div>
  )
} 