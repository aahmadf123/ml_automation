// Types for notification analytics
export type NotificationData = {
  id: string
  title: string
  type: "success" | "error" | "info" | "warning"
  priority: "critical" | "high" | "medium" | "low"
  source: string
  group?: string
  timestamp: Date
  resolved: boolean
  resolvedAt?: Date
  timeToResolve?: number // in minutes
}

// Sources for notifications
const sources = [
  "Data Validation",
  "ML Pipeline",
  "System Admin",
  "External Services",
  "Database",
  "API Gateway",
  "User Activity",
  "Security",
  "Monitoring",
]

// Groups for notifications
const groups = [
  "Data Quality Issues",
  "System Alerts",
  "System Notifications",
  "Security Alerts",
  "Performance Issues",
  "User Feedback",
]

// Generate random notification data
export function generateNotificationData(days = 30, count = 500): NotificationData[] {
  const now = new Date()
  const notifications: NotificationData[] = []

  for (let i = 0; i < count; i++) {
    // Random date within the specified number of days
    const timestamp = new Date(now.getTime() - Math.random() * days * 24 * 60 * 60 * 1000)

    // Determine type with weighted distribution
    const typeRand = Math.random()
    let type: "success" | "error" | "info" | "warning"
    if (typeRand < 0.3) type = "error"
    else if (typeRand < 0.5) type = "warning"
    else if (typeRand < 0.8) type = "info"
    else type = "success"

    // Determine priority with weighted distribution
    const priorityRand = Math.random()
    let priority: "critical" | "high" | "medium" | "low"
    if (priorityRand < 0.05)
      priority = "critical" // 5% critical
    else if (priorityRand < 0.2)
      priority = "high" // 15% high
    else if (priorityRand < 0.5)
      priority = "medium" // 30% medium
    else priority = "low" // 50% low

    // Random source and group
    const source = sources[Math.floor(Math.random() * sources.length)]
    const group = Math.random() > 0.3 ? groups[Math.floor(Math.random() * groups.length)] : undefined

    // Determine if resolved
    const resolved = Math.random() > 0.3

    // For resolved notifications, add resolution time
    let resolvedAt
    let timeToResolve

    if (resolved) {
      // Resolution time depends on priority
      let maxResolutionTime
      switch (priority) {
        case "critical":
          maxResolutionTime = 60 // Max 1 hour for critical
          break
        case "high":
          maxResolutionTime = 240 // Max 4 hours for high
          break
        case "medium":
          maxResolutionTime = 1440 // Max 24 hours for medium
          break
        case "low":
          maxResolutionTime = 4320 // Max 3 days for low
          break
      }

      timeToResolve = Math.floor(Math.random() * maxResolutionTime)
      resolvedAt = new Date(timestamp.getTime() + timeToResolve * 60 * 1000)
    }

    notifications.push({
      id: `notification-${i}`,
      title: `${type.charAt(0).toUpperCase() + type.slice(1)} notification ${i}`,
      type,
      priority,
      source,
      group,
      timestamp,
      resolved,
      resolvedAt,
      timeToResolve,
    })
  }

  // Sort by timestamp (newest first)
  return notifications.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
}

// Function to get notification counts by priority
export function getNotificationCountsByPriority(data: NotificationData[]) {
  return {
    critical: data.filter((n) => n.priority === "critical").length,
    high: data.filter((n) => n.priority === "high").length,
    medium: data.filter((n) => n.priority === "medium").length,
    low: data.filter((n) => n.priority === "low").length,
    total: data.length,
  }
}

// Function to get notification counts by type
export function getNotificationCountsByType(data: NotificationData[]) {
  return {
    error: data.filter((n) => n.type === "error").length,
    warning: data.filter((n) => n.type === "warning").length,
    info: data.filter((n) => n.type === "info").length,
    success: data.filter((n) => n.type === "success").length,
    total: data.length,
  }
}

// Function to get notification counts by source
export function getNotificationCountsBySource(data: NotificationData[]) {
  const counts: Record<string, number> = {}

  data.forEach((notification) => {
    if (!counts[notification.source]) {
      counts[notification.source] = 0
    }
    counts[notification.source]++
  })

  return counts
}

// Function to get notification counts by group
export function getNotificationCountsByGroup(data: NotificationData[]) {
  const counts: Record<string, number> = {
    Ungrouped: 0,
  }

  data.forEach((notification) => {
    if (notification.group) {
      if (!counts[notification.group]) {
        counts[notification.group] = 0
      }
      counts[notification.group]++
    } else {
      counts["Ungrouped"]++
    }
  })

  return counts
}

// Function to get average resolution time by priority
export function getAverageResolutionTimeByPriority(data: NotificationData[]) {
  const result = {
    critical: 0,
    high: 0,
    medium: 0,
    low: 0,
  }

  const counts = {
    critical: 0,
    high: 0,
    medium: 0,
    low: 0,
  }

  data.forEach((notification) => {
    if (notification.resolved && notification.timeToResolve !== undefined) {
      result[notification.priority] += notification.timeToResolve
      counts[notification.priority]++
    }
  })

  // Calculate averages
  Object.keys(result).forEach((key) => {
    const typedKey = key as keyof typeof result
    if (counts[typedKey] > 0) {
      result[typedKey] = Math.round(result[typedKey] / counts[typedKey])
    }
  })

  return result
}

// Function to get notification trends over time
export function getNotificationTrendsByDay(data: NotificationData[], days = 30) {
  const now = new Date()
  const result: Array<{ date: string; critical: number; high: number; medium: number; low: number; total: number }> = []

  // Initialize the result array with dates
  for (let i = 0; i < days; i++) {
    const date = new Date(now.getTime() - (days - i - 1) * 24 * 60 * 60 * 1000)
    const dateString = date.toISOString().split("T")[0]

    result.push({
      date: dateString,
      critical: 0,
      high: 0,
      medium: 0,
      low: 0,
      total: 0,
    })
  }

  // Count notifications for each day
  data.forEach((notification) => {
    const dateString = notification.timestamp.toISOString().split("T")[0]
    const dayData = result.find((d) => d.date === dateString)

    if (dayData) {
      dayData[notification.priority]++
      dayData.total++
    }
  })

  return result
}
