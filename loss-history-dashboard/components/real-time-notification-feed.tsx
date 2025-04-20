"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { useWebSocketContext } from "@/components/websocket-provider"
import { AlertCircle, AlertTriangle, Bell, CheckCircle, Info } from "lucide-react"
import { cn } from "@/lib/utils"
import { motion, AnimatePresence } from "framer-motion"

interface NotificationItem {
  id: string
  title: string
  description?: string
  type: "success" | "error" | "warning" | "info"
  priority: "critical" | "high" | "medium" | "low"
  timestamp: Date
  source: string
}

export function RealTimeNotificationFeed() {
  const [notifications, setNotifications] = useState<NotificationItem[]>([])
  const { lastMessage, messageHistory } = useWebSocketContext()

  // Process incoming WebSocket messages
  useEffect(() => {
    if (lastMessage && lastMessage.type === "notification") {
      const newNotification: NotificationItem = {
        id: lastMessage.id || `notification-${Date.now()}`,
        title: lastMessage.data.title || "New Notification",
        description: lastMessage.data.description,
        type: lastMessage.data.type || "info",
        priority: lastMessage.data.priority || "low",
        timestamp: new Date(lastMessage.timestamp || Date.now()),
        source: lastMessage.data.source || "System",
      }

      setNotifications((prev) => [newNotification, ...prev].slice(0, 10)) // Keep only the 10 most recent
    }
  }, [lastMessage])

  // Initialize with any existing notification messages from history
  useEffect(() => {
    const existingNotifications = messageHistory
      .filter((msg) => msg.type === "notification")
      .map((msg) => ({
        id: msg.id || `notification-${Date.now()}`,
        title: msg.data.title || "Notification",
        description: msg.data.description,
        type: msg.data.type || "info",
        priority: msg.data.priority || "low",
        timestamp: new Date(msg.timestamp || Date.now()),
        source: msg.data.source || "System",
      }))
      .slice(0, 10)

    if (existingNotifications.length > 0) {
      setNotifications(existingNotifications)
    }
  }, [messageHistory])

  const getIcon = (type: string) => {
    switch (type) {
      case "success":
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case "error":
        return <AlertCircle className="h-4 w-4 text-red-500" />
      case "warning":
        return <AlertTriangle className="h-4 w-4 text-yellow-500" />
      case "info":
      default:
        return <Info className="h-4 w-4 text-blue-500" />
    }
  }

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case "critical":
        return "border-purple-500 bg-purple-50 dark:bg-purple-900/20"
      case "high":
        return "border-red-500 bg-red-50 dark:bg-red-900/20"
      case "medium":
        return "border-amber-500 bg-amber-50 dark:bg-amber-900/20"
      case "low":
        return "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
      default:
        return "border-gray-200 bg-gray-50 dark:bg-gray-800/20"
    }
  }

  const getPriorityBadge = (priority: string) => {
    switch (priority) {
      case "critical":
        return (
          <Badge
            variant="outline"
            className="bg-purple-100 text-purple-800 border-purple-300 dark:bg-purple-900/30 dark:text-purple-200"
          >
            critical
          </Badge>
        )
      case "high":
        return (
          <Badge
            variant="outline"
            className="bg-red-100 text-red-800 border-red-300 dark:bg-red-900/30 dark:text-red-200"
          >
            high
          </Badge>
        )
      case "medium":
        return (
          <Badge
            variant="outline"
            className="bg-amber-100 text-amber-800 border-amber-300 dark:bg-amber-900/30 dark:text-amber-200"
          >
            medium
          </Badge>
        )
      case "low":
        return (
          <Badge
            variant="outline"
            className="bg-blue-100 text-blue-800 border-blue-300 dark:bg-blue-900/30 dark:text-blue-200"
          >
            low
          </Badge>
        )
      default:
        return null
    }
  }

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-lg flex items-center gap-2">
          <Bell className="h-5 w-5" />
          Real-Time Notifications
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[400px] pr-4">
          <AnimatePresence>
            {notifications.length > 0 ? (
              <div className="space-y-2">
                {notifications.map((notification) => (
                  <motion.div
                    key={notification.id}
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.3 }}
                    className={cn("p-3 rounded-md border-l-4 shadow-sm", getPriorityColor(notification.priority))}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex items-center gap-2">
                        {getIcon(notification.type)}
                        <span className="font-medium">{notification.title}</span>
                        {getPriorityBadge(notification.priority)}
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">
                        {formatTime(notification.timestamp)}
                      </div>
                    </div>
                    {notification.description && (
                      <div className="mt-1 text-sm text-gray-600 dark:text-gray-300">{notification.description}</div>
                    )}
                    <div className="mt-1 text-xs text-gray-500 dark:text-gray-400">Source: {notification.source}</div>
                  </motion.div>
                ))}
              </div>
            ) : (
              <div className="flex h-[300px] items-center justify-center text-gray-500 dark:text-gray-400">
                <p>No notifications yet. They will appear here as they arrive.</p>
              </div>
            )}
          </AnimatePresence>
        </ScrollArea>
      </CardContent>
    </Card>
  )
}
