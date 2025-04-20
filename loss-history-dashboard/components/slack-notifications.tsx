"use client"

import { useState, useEffect } from "react"
import { useWebSocketContext } from "@/components/websocket-provider"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Bell, MessageSquare, CheckCircle, AlertCircle, Clock, Send, Settings, Filter } from "lucide-react"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

// Define types for our notification data
interface SlackNotification {
  id: string
  title: string
  message: string
  type: "info" | "warning" | "error" | "success"
  priority: "low" | "medium" | "high" | "critical"
  source: string
  timestamp: Date
  status: "sent" | "pending" | "failed"
  channel?: string
}

// Mock data for initial rendering
const initialNotifications: SlackNotification[] = [
  {
    id: "notif1",
    title: "Data validation error",
    message: "Invalid property values detected in recent import",
    type: "error",
    priority: "high",
    source: "Data Ingestion",
    timestamp: new Date(Date.now() - 3600000),
    status: "sent",
    channel: "ml-alerts"
  },
  {
    id: "notif2",
    title: "Model drift detected",
    message: "Feature 'claim_amount' has drifted beyond threshold",
    type: "warning",
    priority: "medium",
    source: "Model Monitoring",
    timestamp: new Date(Date.now() - 7200000),
    status: "sent",
    channel: "ml-alerts"
  },
  {
    id: "notif3",
    title: "System update completed",
    message: "System has been updated to version 2.3.1",
    type: "success",
    priority: "low",
    source: "System",
    timestamp: new Date(Date.now() - 10800000),
    status: "sent",
    channel: "ml-system"
  },
  {
    id: "notif4",
    title: "Critical service outage",
    message: "External API service is currently unavailable",
    type: "error",
    priority: "critical",
    source: "External Services",
    timestamp: new Date(Date.now() - 14400000),
    status: "sent",
    channel: "ml-critical"
  },
  {
    id: "notif5",
    title: "Model training completed",
    message: "Gradient Boosting model training completed with RMSE: 0.26",
    type: "success",
    priority: "medium",
    source: "Model Training",
    timestamp: new Date(Date.now() - 18000000),
    status: "sent",
    channel: "ml-training"
  }
]

export function SlackNotifications() {
  const { lastMessage } = useWebSocketContext()
  const [notifications, setNotifications] = useState<SlackNotification[]>(initialNotifications)
  const [activeTab, setActiveTab] = useState("all")
  const [notificationsEnabled, setNotificationsEnabled] = useState(true)
  const [selectedChannel, setSelectedChannel] = useState("all")
  const [selectedPriority, setSelectedPriority] = useState("all")

  // Process incoming WebSocket messages
  useEffect(() => {
    if (!lastMessage) return

    // Handle new notification
    if (lastMessage.type === "notification") {
      const newNotification: SlackNotification = {
        id: Date.now().toString(),
        title: lastMessage.data.title,
        message: lastMessage.data.description,
        type: lastMessage.data.type,
        priority: lastMessage.data.priority,
        source: lastMessage.data.source,
        timestamp: new Date(),
        status: "pending",
        channel: lastMessage.data.channel || "ml-alerts"
      }
      
      setNotifications(prev => [newNotification, ...prev])
      
      // Simulate sending the notification after a short delay
      setTimeout(() => {
        setNotifications(prev => 
          prev.map(notif => 
            notif.id === newNotification.id 
              ? { ...notif, status: "sent" } 
              : notif
          )
        )
      }, 2000)
    }
  }, [lastMessage])

  // Filter notifications based on active tab and filters
  const filteredNotifications = notifications.filter(notification => {
    // Filter by tab
    if (activeTab !== "all" && notification.type !== activeTab) {
      return false
    }
    
    // Filter by channel
    if (selectedChannel !== "all" && notification.channel !== selectedChannel) {
      return false
    }
    
    // Filter by priority
    if (selectedPriority !== "all" && notification.priority !== selectedPriority) {
      return false
    }
    
    return true
  })

  // Get notification count by type
  const notificationCounts = {
    all: notifications.length,
    info: notifications.filter(n => n.type === "info").length,
    warning: notifications.filter(n => n.type === "warning").length,
    error: notifications.filter(n => n.type === "error").length,
    success: notifications.filter(n => n.type === "success").length
  }

  // Helper function to get notification icon
  const getNotificationIcon = (type: string) => {
    switch (type) {
      case "success":
        return <CheckCircle className="h-4 w-4 text-emerald-600" />
      case "warning":
        return <AlertCircle className="h-4 w-4 text-amber-600" />
      case "error":
        return <AlertCircle className="h-4 w-4 text-rose-600" />
      default:
        return <Bell className="h-4 w-4 text-sky-600" />
    }
  }

  // Helper function to get priority badge color
  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case "critical":
        return "bg-rose-500/15 text-rose-600"
      case "high":
        return "bg-orange-500/15 text-orange-600"
      case "medium":
        return "bg-amber-500/15 text-amber-600"
      case "low":
        return "bg-emerald-500/15 text-emerald-600"
      default:
        return "bg-gray-500/15 text-gray-600"
    }
  }

  // Helper function to get status badge color
  const getStatusColor = (status: string) => {
    switch (status) {
      case "sent":
        return "bg-emerald-500/15 text-emerald-600"
      case "pending":
        return "bg-amber-500/15 text-amber-600"
      case "failed":
        return "bg-rose-500/15 text-rose-600"
      default:
        return "bg-gray-500/15 text-gray-600"
    }
  }

  // Helper function to get status icon
  const getStatusIcon = (status: string) => {
    switch (status) {
      case "sent":
        return <CheckCircle className="h-4 w-4" />
      case "pending":
        return <Clock className="h-4 w-4" />
      case "failed":
        return <AlertCircle className="h-4 w-4" />
      default:
        return <Clock className="h-4 w-4" />
    }
  }

  return (
    <div className="space-y-6 bg-gradient-to-b from-gray-50 to-gray-100 p-6 rounded-lg">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-800">Slack Notifications</h2>
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <Switch 
              id="notifications-enabled" 
              checked={notificationsEnabled} 
              onCheckedChange={setNotificationsEnabled} 
            />
            <Label htmlFor="notifications-enabled" className="text-gray-700">Enable Notifications</Label>
          </div>
          <Button variant="outline" size="sm" className="border-gray-300 hover:bg-gray-100">
            <Settings className="h-4 w-4 mr-2 text-gray-600" />
            Settings
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="bg-white shadow-sm border border-gray-200">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-700">Total Notifications</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-gray-800">{notificationCounts.all}</div>
            <p className="text-xs text-gray-500">Last 24 hours</p>
          </CardContent>
        </Card>
        <Card className="bg-white shadow-sm border border-gray-200">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-700">Critical Alerts</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-rose-600">
              {notifications.filter(n => n.priority === "critical").length}
            </div>
            <p className="text-xs text-gray-500">Requires immediate attention</p>
          </CardContent>
        </Card>
        <Card className="bg-white shadow-sm border border-gray-200">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-700">Delivery Rate</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-gray-800">
              {Math.round((notifications.filter(n => n.status === "sent").length / notifications.length) * 100)}%
            </div>
            <p className="text-xs text-gray-500">Successfully delivered</p>
          </CardContent>
        </Card>
      </div>

      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full md:w-auto">
          <TabsList>
            <TabsTrigger value="all" className="relative">
              All
              <Badge variant="secondary" className="ml-2 h-5 w-5 rounded-full p-0 flex items-center justify-center">
                {notificationCounts.all}
              </Badge>
            </TabsTrigger>
            <TabsTrigger value="info" className="relative">
              Info
              <Badge variant="secondary" className="ml-2 h-5 w-5 rounded-full p-0 flex items-center justify-center">
                {notificationCounts.info}
              </Badge>
            </TabsTrigger>
            <TabsTrigger value="warning" className="relative">
              Warning
              <Badge variant="secondary" className="ml-2 h-5 w-5 rounded-full p-0 flex items-center justify-center">
                {notificationCounts.warning}
              </Badge>
            </TabsTrigger>
            <TabsTrigger value="error" className="relative">
              Error
              <Badge variant="secondary" className="ml-2 h-5 w-5 rounded-full p-0 flex items-center justify-center">
                {notificationCounts.error}
              </Badge>
            </TabsTrigger>
            <TabsTrigger value="success" className="relative">
              Success
              <Badge variant="secondary" className="ml-2 h-5 w-5 rounded-full p-0 flex items-center justify-center">
                {notificationCounts.success}
              </Badge>
            </TabsTrigger>
          </TabsList>
        </Tabs>

        <div className="flex flex-col sm:flex-row gap-2">
          <Select value={selectedChannel} onValueChange={setSelectedChannel}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Filter by channel" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Channels</SelectItem>
              <SelectItem value="ml-alerts">ML Alerts</SelectItem>
              <SelectItem value="ml-system">ML System</SelectItem>
              <SelectItem value="ml-critical">ML Critical</SelectItem>
              <SelectItem value="ml-training">ML Training</SelectItem>
            </SelectContent>
          </Select>

          <Select value={selectedPriority} onValueChange={setSelectedPriority}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Filter by priority" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Priorities</SelectItem>
              <SelectItem value="critical">Critical</SelectItem>
              <SelectItem value="high">High</SelectItem>
              <SelectItem value="medium">Medium</SelectItem>
              <SelectItem value="low">Low</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <Card className="bg-white shadow-sm border border-gray-200">
        <CardHeader>
          <CardTitle className="text-gray-800">Notification History</CardTitle>
          <CardDescription className="text-gray-500">Recent Slack notifications sent from the ML pipeline</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {filteredNotifications.length === 0 ? (
              <div className="text-center py-8">
                <MessageSquare className="h-12 w-12 mx-auto text-gray-400" />
                <p className="mt-2 text-sm text-gray-500">No notifications match the current filters</p>
              </div>
            ) : (
              filteredNotifications.map(notification => (
                <div key={notification.id} className="border border-gray-200 rounded-lg p-4 space-y-2 bg-white shadow-sm">
                  <div className="flex items-start justify-between">
                    <div className="flex items-start space-x-3">
                      <div className="mt-1">
                        {getNotificationIcon(notification.type)}
                      </div>
                      <div>
                        <h3 className="font-medium text-gray-800">{notification.title}</h3>
                        <p className="text-sm text-gray-600">{notification.message}</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Badge className={getPriorityColor(notification.priority)}>
                        {notification.priority}
                      </Badge>
                      <Badge className={getStatusColor(notification.status)}>
                        {getStatusIcon(notification.status)}
                        <span className="ml-1">{notification.status}</span>
                      </Badge>
                    </div>
                  </div>
                  <div className="flex items-center justify-between text-xs text-gray-500">
                    <div className="flex items-center space-x-2">
                      <span>Source: {notification.source}</span>
                      {notification.channel && (
                        <>
                          <span>•</span>
                          <span>Channel: {notification.channel}</span>
                        </>
                      )}
                    </div>
                    <div>
                      {notification.timestamp.toLocaleString()}
                    </div>
                  </div>
                  {notification.status === "pending" && (
                    <div className="mt-2">
                      <div className="w-full bg-gray-200 rounded-full h-1.5">
                        <div className="bg-sky-500 h-1.5 rounded-full w-1/3 animate-pulse"></div>
                      </div>
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        </CardContent>
      </Card>

      <Card className="bg-white shadow-sm border border-gray-200">
        <CardHeader>
          <CardTitle className="text-gray-800">Notification Settings</CardTitle>
          <CardDescription className="text-gray-500">Configure which events trigger Slack notifications</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-medium text-gray-800">Data Drift Alerts</h3>
                <p className="text-sm text-gray-600">Send notifications when data drift is detected</p>
              </div>
              <Switch defaultChecked />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-medium text-gray-800">Model Performance Alerts</h3>
                <p className="text-sm text-gray-600">Send notifications when model metrics degrade</p>
              </div>
              <Switch defaultChecked />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-medium text-gray-800">Pipeline Status Updates</h3>
                <p className="text-sm text-gray-600">Send notifications for pipeline stage transitions</p>
              </div>
              <Switch defaultChecked />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-medium text-gray-800">System Resource Alerts</h3>
                <p className="text-sm text-gray-600">Send notifications for high resource usage</p>
              </div>
              <Switch defaultChecked />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-medium text-gray-800">Training Completion</h3>
                <p className="text-sm text-gray-600">Send notifications when model training completes</p>
              </div>
              <Switch defaultChecked />
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
} 