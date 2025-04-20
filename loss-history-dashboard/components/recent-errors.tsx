"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { useWebSocketContext } from "@/components/websocket-provider"
import { formatDistanceToNow } from "date-fns"
import { RefreshCw } from "lucide-react"

interface ErrorItem {
  id: number
  message: string
  timestamp: string
  severity: "error" | "warning" | "info"
}

// Helper function to safely format dates
const safeFormatDistanceToNow = (dateString: string): string => {
  try {
    // First check if the date is valid
    const date = new Date(dateString)
    if (isNaN(date.getTime())) {
      return "Invalid date"
    }
    return formatDistanceToNow(date, { addSuffix: true })
  } catch (error) {
    console.error("Error formatting date:", error)
    return "Unknown time"
  }
}

export function RecentErrors() {
  const { lastMessage } = useWebSocketContext()
  const [errors, setErrors] = useState<ErrorItem[]>([
    {
      id: 1,
      message: "Schema mismatch in column 'property_value'",
      timestamp: new Date(Date.now() - 3600000).toISOString(),
      severity: "warning",
    },
    {
      id: 2,
      message: "Missing values in 'year_built' exceed threshold",
      timestamp: new Date(Date.now() - 10800000).toISOString(),
      severity: "error",
    },
    {
      id: 3,
      message: "Feature drift detected in 'claim_amount'",
      timestamp: new Date(Date.now() - 18000000).toISOString(),
      severity: "warning",
    },
    {
      id: 4,
      message: "Outlier detected in 'location_risk' feature",
      timestamp: new Date(Date.now() - 25200000).toISOString(),
      severity: "info",
    },
    {
      id: 5,
      message: "Data quality score below threshold",
      timestamp: new Date(Date.now() - 32400000).toISOString(),
      severity: "error",
    },
  ])
  const [isUpdating, setIsUpdating] = useState(false)
  const [newErrorId, setNewErrorId] = useState<number | null>(null)

  // Update errors when a new error alert message is received
  useEffect(() => {
    if (lastMessage && lastMessage.type === "error_alert") {
      setIsUpdating(true)

      // Ensure we have a valid timestamp
      let timestamp = lastMessage.data.timestamp
      if (!timestamp) {
        timestamp = new Date().toISOString()
      }

      const newError: ErrorItem = {
        id: lastMessage.data.id,
        message: lastMessage.data.message,
        timestamp: timestamp,
        severity: lastMessage.data.severity,
      }

      // Add the new error and keep only the most recent 10
      setErrors((prev) => [newError, ...prev].slice(0, 10))
      setNewErrorId(newError.id)

      setTimeout(() => {
        setIsUpdating(false)
        // Clear the highlight after animation
        setTimeout(() => setNewErrorId(null), 3000)
      }, 500)
    }
  }, [lastMessage])

  return (
    <Card className={`transition-all duration-300 ${isUpdating ? "border-primary" : ""}`}>
      <CardHeader>
        <CardTitle className="text-lg flex items-center justify-between">
          Recent Errors
          {isUpdating && <RefreshCw className="h-4 w-4 animate-spin text-primary" />}
        </CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <div className="h-[250px] overflow-y-auto px-6 py-2 scrollbar-thin scrollbar-thumb-rounded scrollbar-thumb-gray-300">
          <div className="space-y-4">
            {errors.map((error) => (
              <div
                key={error.id}
                className={`flex items-start justify-between transition-all duration-500 ${
                  newErrorId === error.id ? "bg-primary/10 -mx-2 px-2 py-1 rounded-md" : ""
                }`}
              >
                <div className="space-y-1">
                  <p className="font-medium">{error.message}</p>
                  <p className="text-sm text-muted-foreground">{safeFormatDistanceToNow(error.timestamp)}</p>
                </div>
                <Badge
                  variant={
                    error.severity === "error" ? "destructive" : error.severity === "warning" ? "warning" : "secondary"
                  }
                >
                  {error.severity}
                </Badge>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
