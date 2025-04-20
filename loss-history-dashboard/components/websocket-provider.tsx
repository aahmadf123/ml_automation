"use client"

import { createContext, useContext, type ReactNode, useState, useEffect } from "react"
import { useToast } from "@/hooks/use-toast"

// Define WebSocket types
export type ConnectionState = "disconnected" | "connecting" | "connected" | "error"

export interface WebSocketMessage {
  id?: string
  timestamp?: Date
  type: string
  data: any
}

// WebSocket Provider Context
interface WebSocketContextType {
  connectionState: ConnectionState
  lastMessage: WebSocketMessage | null
  messageHistory: WebSocketMessage[]
  isPaused: boolean
  connect: () => void
  disconnect: () => void
  sendMessage: (message: any) => boolean
  togglePause: () => void
  clearMessages: () => void
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined)

export function useWebSocketContext(): WebSocketContextType {
  const context = useContext(WebSocketContext)
  if (!context) {
    // Return a mock implementation instead of throwing an error
    return {
      connectionState: "connected",
      lastMessage: null,
      messageHistory: [],
      isPaused: false,
      connect: () => {},
      disconnect: () => {},
      sendMessage: () => true,
      togglePause: () => {},
      clearMessages: () => {},
    }
  }
  return context
}

interface WebSocketProviderProps {
  children: ReactNode
}

// Mock data generator for testing UI
const generateMockData = () => {
  const mockMessages: WebSocketMessage[] = [
    {
      type: "drift_alert",
      data: {
        feature: "property_value",
        current: 3.2,
        threshold: 5.0,
        status: "warning",
        history: [1.5, 2.1, 2.7, 3.0, 3.2],
      },
    },
    {
      type: "model_metrics",
      data: {
        model: "model1",
        metrics: {
          rmse: "0.26",
          mae: "0.22",
          mse: "0.068",
          r2: "0.88",
        },
      },
    },
    {
      type: "error_alert",
      data: {
        severity: "warning",
        message: "Data quality check failed for recent imports",
      },
    },
    // Add notification messages
    {
      type: "notification",
      data: {
        title: "Data validation error",
        description: "Invalid property values detected in recent import",
        type: "error",
        priority: "high",
        source: "Data Ingestion",
      },
    },
    {
      type: "notification",
      data: {
        title: "Model drift detected",
        description: "Feature 'claim_amount' has drifted beyond threshold",
        type: "warning",
        priority: "medium",
        source: "Model Monitoring",
      },
    },
    {
      type: "notification",
      data: {
        title: "System update completed",
        description: "System has been updated to version 2.3.1",
        type: "success",
        priority: "low",
        source: "System",
      },
    },
    {
      type: "notification",
      data: {
        title: "Critical service outage",
        description: "External API service is currently unavailable",
        type: "error",
        priority: "critical",
        source: "External Services",
      },
    },
  ]

  return mockMessages[Math.floor(Math.random() * mockMessages.length)]
}

export function WebsocketProvider({ children }: WebSocketProviderProps) {
  const { toast } = useToast()
  const [connectionState, setConnectionState] = useState<ConnectionState>("connected")
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null)
  const [messageHistory, setMessageHistory] = useState<WebSocketMessage[]>([])
  const [isPaused, setIsPaused] = useState(false)

  // For demo purposes, we'll simulate receiving messages periodically
  useEffect(() => {
    // Don't send mock messages if paused
    if (isPaused) return

    const interval = setInterval(() => {
      const mockMessage = generateMockData()
      mockMessage.timestamp = new Date()
      mockMessage.id = Date.now().toString()

      setLastMessage(mockMessage)
      setMessageHistory((prev) => [...prev, mockMessage])

      // Show toast notifications for important messages
      if (mockMessage.type === "error_alert" && mockMessage.data.severity === "error") {
        toast({
          title: "Error Alert",
          description: mockMessage.data.message,
          variant: "destructive",
        })
      }

      // Show toast for critical notifications
      if (mockMessage.type === "notification" && mockMessage.data.priority === "critical") {
        toast.error(mockMessage.data.title, {
          description: mockMessage.data.description,
          persistent: true,
          priority: "critical",
        })
      }
    }, 8000) // Send a mock message every 8 seconds (increased frequency)

    return () => clearInterval(interval)
  }, [isPaused, toast])

  const connect = () => setConnectionState("connected")
  const disconnect = () => setConnectionState("disconnected")
  const sendMessage = () => true // Mock implementation
  const togglePause = () => setIsPaused((prev) => !prev)
  const clearMessages = () => {
    setMessageHistory([])
    setLastMessage(null)
  }

  const value = {
    connectionState,
    lastMessage,
    messageHistory,
    isPaused,
    connect,
    disconnect,
    sendMessage,
    togglePause,
    clearMessages,
  }

  return <WebSocketContext.Provider value={value}>{children}</WebSocketContext.Provider>
}
