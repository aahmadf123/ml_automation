"use client"

import { useState, useEffect, useCallback } from "react"

// WebSocket message types
export type WebSocketMessage = {
  id: string
  type: string
  timestamp: string
  data: any
}

// WebSocket connection states
export type ConnectionState = "connecting" | "open" | "closed" | "error"

// WebSocket configuration
export interface WebSocketConfig {
  url: string
}

export function useWebSocket(url: string = "ws://localhost:8000") {
  const [socket, setSocket] = useState<WebSocket | null>(null)
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null)
  const [isConnected, setIsConnected] = useState(false)

  useEffect(() => {
    const ws = new WebSocket(url)

    ws.onopen = () => {
      setIsConnected(true)
      console.log("WebSocket connected")
    }

    ws.onclose = () => {
      setIsConnected(false)
      console.log("WebSocket disconnected")
    }

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data)
        setLastMessage({
          ...message,
          timestamp: Date.now(),
        })
      } catch (error) {
        console.error("Failed to parse WebSocket message:", error)
      }
    }

    ws.onerror = (error) => {
      console.error("WebSocket error:", error)
    }

    setSocket(ws)

    return () => {
      ws.close()
    }
  }, [url])

  const sendMessage = useCallback(
    (message: any) => {
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify(message))
      } else {
        console.warn("WebSocket is not connected")
      }
    },
    [socket]
  )

  return {
    lastMessage,
    isConnected,
    sendMessage,
  }
}
