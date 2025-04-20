import { Card } from "@/components/ui/card"

interface SystemMetricsProps {
  metrics: {
    runtime: number
    memoryUsage: number
    cpuUsage: number
    gpuUsage?: number
  }
}

export function SystemMetrics({ metrics }: SystemMetricsProps) {
  const formatMemory = (bytes: number) => {
    const mb = bytes / (1024 * 1024)
    return `${mb.toFixed(2)} MB`
  }

  const formatPercentage = (value: number) => {
    return `${value.toFixed(1)}%`
  }

  const formatRuntime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    const remainingSeconds = Math.floor(seconds % 60)
    return `${hours}h ${minutes}m ${remainingSeconds}s`
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      <Card
        title="Runtime"
        value={formatRuntime(metrics.runtime)}
      />
      <Card
        title="Memory Usage"
        value={formatMemory(metrics.memoryUsage)}
      />
      <Card
        title="CPU Usage"
        value={formatPercentage(metrics.cpuUsage)}
      />
      {metrics.gpuUsage !== undefined && (
        <Card
          title="GPU Usage"
          value={formatPercentage(metrics.gpuUsage)}
        />
      )}
    </div>
  )
} 