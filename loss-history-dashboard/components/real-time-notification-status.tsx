"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useWebSocketContext } from "@/components/websocket-provider"
import { Badge } from "@/components/ui/badge"
import { Activity, CheckCircle, Clock, XCircle } from "lucide-react"

export function RealTimeNotificationStatus() {
  const [status, setStatus] = useState({
    active: true,
    lastReceived: new Date(),
    messageCount: 0,
    connectionUptime: 0,
  })
  const { connectionState, lastMessage, messageHistory } = useWebSocketContext()
  const [uptimeInterval, setUptimeInterval] = useState<NodeJS.Timeout | null>(null)

  // Update status when connection state changes
  useEffect(() => {
    setStatus((prev) => ({
      ...prev,
      active: connectionState === "connected",
    }))

    // Start uptime counter when connected
    if (connectionState === "connected" && !uptimeInterval) {
      const interval = setInterval(() => {
        setStatus((prev) => ({
          ...prev,
          connectionUptime: prev.connectionUptime + 1,
        }))
      }, 1000)

      setUptimeInterval(interval)
    } else if (connectionState !== "connected" && uptimeInterval) {
      clearInterval(uptimeInterval)
      setUptimeInterval(null)
    }

    return () => {
      if (uptimeInterval) {
        clearInterval(uptimeInterval)
      }
    }
  }, [connectionState, uptimeInterval])

  // Update status when new messages arrive
  useEffect(() => {
    if (lastMessage) {
      setStatus((prev) => ({
        ...prev,
        lastReceived: new Date(),
        messageCount: messageHistory.length,
      }))
    }
  }, [lastMessage, messageHistory])

  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    const secs = seconds % 60

    return `${hours.toString().padStart(2, "0")}:${minutes.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`
  }

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-lg flex items-center gap-2">
          <Activity className="h-5 w-5" />
          WebSocket Status
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4">
          <div className="flex flex-col gap-1">
            <div className="text-sm font-medium text-gray-500 dark:text-gray-400">Connection Status</div>
            <div className="flex items-center gap-2">
              {status.active ? (
                <>
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <Badge
                    variant="outline"
                    className="bg-green-100 text-green-800 border-green-300 dark:bg-green-900/30 dark:text-green-200"
                  >
                    Connected
                  </Badge>
                </>
              ) : (
                <>
                  <XCircle className="h-4 w-4 text-red-500" />
                  <Badge
                    variant="outline"
                    className="bg-red-100 text-red-800 border-red-300 dark:bg-red-900/30 dark:text-red-200"
                  >
                    Disconnected
                  </Badge>
                </>
              )}
            </div>
          </div>

          <div className="flex flex-col gap-1">
            <div className="text-sm font-medium text-gray-500 dark:text-gray-400">Connection Uptime</div>
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4 text-blue-500" />
              <span className="font-mono">{formatUptime(status.connectionUptime)}</span>
            </div>
          </div>

          <div className="flex flex-col gap-1">
            <div className="text-sm font-medium text-gray-500 dark:text-gray-400">Last Message</div>
            <div className="font-mono">{formatTime(status.lastReceived)}</div>
          </div>

          <div className="flex flex-col gap-1">
            <div className="text-sm font-medium text-gray-500 dark:text-gray-400">Total Messages</div>
            <div className="font-mono">{status.messageCount}</div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
