"use client"

import type React from "react"

import { toast as sonnerToast, Toaster as SonnerToaster } from "sonner"

// Update the ExtendedToastOptions type to include priority
type ExtendedToastOptions = Parameters<typeof sonnerToast>[1] & {
  persistent?: boolean
  group?: string // Group identifier
  source?: string // Source component/system
  priority?: "low" | "medium" | "high" | "critical" // Priority level
}

// Update the PersistentNotification type to include priority
type PersistentNotification = {
  id: string
  title: string
  description?: string
  type?: "success" | "error" | "info" | "warning"
  action?: {
    label: string
    onClick: () => void
  }
  group?: string
  source?: string
  priority?: "low" | "medium" | "high" | "critical" // Priority level
  timestamp: number
}

// Update the NotificationGroup type
type NotificationGroup = {
  id: string
  title: string
  type: string
  count: number
  notifications: PersistentNotification[]
  timestamp: number
  priority: number // Add priority field
}

// Store for persistent notifications
let persistentNotifications: PersistentNotification[] = []
let listeners: (() => void)[] = []

// Function to notify listeners when notifications change
const notifyListeners = () => {
  listeners.forEach((listener) => listener())
}

// Update the groupNotifications function to sort by priority first, then timestamp
const groupNotifications = (notifications: PersistentNotification[]): NotificationGroup[] => {
  const groups: Record<string, NotificationGroup> = {}

  // First pass: handle explicit groups
  notifications.forEach((notification) => {
    if (notification.group) {
      if (!groups[notification.group]) {
        groups[notification.group] = {
          id: notification.group,
          title: notification.group,
          type: notification.type || "info",
          count: 0,
          notifications: [],
          timestamp: notification.timestamp,
          priority: getPriorityValue(notification.priority), // Store highest priority value for the group
        }
      }

      groups[notification.group].notifications.push(notification)
      groups[notification.group].count++

      // Update timestamp to the most recent
      if (notification.timestamp > groups[notification.group].timestamp) {
        groups[notification.group].timestamp = notification.timestamp
      }

      // Update group priority to highest priority in the group
      const notificationPriority = getPriorityValue(notification.priority)
      if (notificationPriority > groups[notification.group].priority) {
        groups[notification.group].priority = notificationPriority
      }
    }
  })

  // Second pass: auto-group by type and source for ungrouped notifications
  notifications.forEach((notification) => {
    if (!notification.group) {
      const autoGroupKey = `${notification.source || "unknown"}-${notification.type || "info"}`

      if (!groups[autoGroupKey]) {
        groups[autoGroupKey] = {
          id: autoGroupKey,
          title: notification.source
            ? `${notification.source} ${notification.type || "notifications"}`
            : `${notification.type || "General"} notifications`,
          type: notification.type || "info",
          count: 0,
          notifications: [],
          timestamp: notification.timestamp,
          priority: getPriorityValue(notification.priority), // Initialize with first notification's priority
        }
      }

      groups[autoGroupKey].notifications.push(notification)
      groups[autoGroupKey].count++

      // Update timestamp to the most recent
      if (notification.timestamp > groups[autoGroupKey].timestamp) {
        groups[autoGroupKey].timestamp = notification.timestamp
      }

      // Update group priority to highest priority in the group
      const notificationPriority = getPriorityValue(notification.priority)
      if (notificationPriority > groups[autoGroupKey].priority) {
        groups[autoGroupKey].priority = notificationPriority
      }
    }
  })

  // Convert to array and sort by priority (highest to lowest) first, then by timestamp (newest first)
  return Object.values(groups).sort((a, b) => {
    // Sort by priority first
    if (a.priority !== b.priority) {
      return b.priority - a.priority
    }
    // Then by timestamp
    return b.timestamp - a.timestamp
  })
}

// Helper function to get numeric value for priority
const getPriorityValue = (priority?: string): number => {
  switch (priority) {
    case "critical":
      return 3
    case "high":
      return 2
    case "medium":
      return 1
    case "low":
    default:
      return 0
  }
}

// Enhanced toast function with persistent option and grouping
const toast = Object.assign(
  (message: string | React.ReactNode, options?: ExtendedToastOptions) => {
    if (options?.persistent) {
      const id = Math.random().toString(36).substring(2, 9)
      persistentNotifications.push({
        id,
        title: typeof message === "string" ? message : "Notification",
        description: options?.description,
        type: options?.type || "info",
        action: options?.action,
        group: options?.group,
        source: options?.source,
        priority: options?.priority || "low", // Add priority with default
        timestamp: Date.now(),
      })
      notifyListeners()
      return id
    }
    return sonnerToast(message, options)
  },
  {
    success: (message: string | React.ReactNode, options?: ExtendedToastOptions) => {
      if (options?.persistent) {
        const id = Math.random().toString(36).substring(2, 9)
        persistentNotifications.push({
          id,
          title: typeof message === "string" ? message : "Success",
          description: options?.description,
          type: "success",
          action: options?.action,
          group: options?.group,
          source: options?.source,
          priority: options?.priority || "low", // Add priority with default
          timestamp: Date.now(),
        })
        notifyListeners()
        return id
      }
      return sonnerToast.success(message, options)
    },
    error: (message: string | React.ReactNode, options?: ExtendedToastOptions) => {
      if (options?.persistent) {
        const id = Math.random().toString(36).substring(2, 9)
        persistentNotifications.push({
          id,
          title: typeof message === "string" ? message : "Error",
          description: options?.description,
          type: "error",
          action: options?.action,
          group: options?.group,
          source: options?.source,
          priority: options?.priority || "medium", // Default to medium for errors
          timestamp: Date.now(),
        })
        notifyListeners()
        return id
      }
      return sonnerToast.error(message, options)
    },
    info: (message: string | React.ReactNode, options?: ExtendedToastOptions) => {
      if (options?.persistent) {
        const id = Math.random().toString(36).substring(2, 9)
        persistentNotifications.push({
          id,
          title: typeof message === "string" ? message : "Info",
          description: options?.description,
          type: "info",
          action: options?.action,
          group: options?.group,
          source: options?.source,
          priority: options?.priority || "low", // Default to low for info
          timestamp: Date.now(),
        })
        notifyListeners()
        return id
      }
      return sonnerToast.info(message, options)
    },
    warning: (message: string | React.ReactNode, options?: ExtendedToastOptions) => {
      if (options?.persistent) {
        const id = Math.random().toString(36).substring(2, 9)
        persistentNotifications.push({
          id,
          title: typeof message === "string" ? message : "Warning",
          description: options?.description,
          type: "warning",
          action: options?.action,
          group: options?.group,
          source: options?.source,
          priority: options?.priority || "low", // Default to low for warnings
          timestamp: Date.now(),
        })
        notifyListeners()
        return id
      }
      return sonnerToast.warning(message, options)
    },
    dismiss: (id?: string) => {
      if (id && persistentNotifications.some((n) => n.id === id)) {
        persistentNotifications = persistentNotifications.filter((n) => n.id !== id)
        notifyListeners()
      } else {
        sonnerToast.dismiss(id)
      }
    },
    // Dismiss all notifications in a group
    dismissGroup: (groupId: string) => {
      persistentNotifications = persistentNotifications.filter((n) => n.group !== groupId)
      notifyListeners()
    },
    // Function to get persistent notifications
    getPersistentNotifications: () => [...persistentNotifications],
    // Function to get grouped notifications
    getGroupedNotifications: () => groupNotifications([...persistentNotifications]),
    // Subscribe to notification changes
    subscribe: (callback: () => void) => {
      listeners.push(callback)
      return () => {
        listeners = listeners.filter((l) => l !== callback)
      }
    },
  },
)

// Export the Toaster component
export function Toaster() {
  return (
    <SonnerToaster
      position="bottom-right"
      toastOptions={{
        className: "bg-background text-foreground border-border",
        descriptionClassName: "text-muted-foreground",
      }}
    />
  )
}

export { toast }
export type { NotificationGroup, PersistentNotification }
