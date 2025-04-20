"use client"

import { useState, useEffect, useRef, useCallback } from "react"

// Define the types if they're not imported from lib/websocket
export type ConnectionState = "disconnected" | "connecting" | "connected" | "error"

export interface WebSocketMessage {
  id?: string
  timestamp?: Date
  type: string
  data: any
}

interface WebSocketOptions {
  url: string
  reconnectInterval?: number
  maxReconnectAttempts?: number
}

interface WebSocketCallbacks {
  onOpen?: (event: Event) => void
  onMessage?: (message: WebSocketMessage) => void
  onClose?: (event: CloseEvent) => void
  onError?: (event: Event) => void
}

export function useWebSocket(options: WebSocketOptions, callbacks?: WebSocketCallbacks) {
  const { url, reconnectInterval = 3000, maxReconnectAttempts = 5 } = options
  const [connectionState, setConnectionState] = useState<ConnectionState>("disconnected")
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null)
  const [messageHistory, setMessageHistory] = useState<WebSocketMessage[]>([])
  const [isPaused, setIsPaused] = useState(false)
  const socketRef = useRef<WebSocket | null>(null)
  const reconnectAttemptsRef = useRef(0)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const pausedRef = useRef(isPaused)

  // Update the ref when isPaused changes
  useEffect(() => {
    pausedRef.current = isPaused
  }, [isPaused])

  const connect = useCallback(() => {
    // Don't connect if we're already connected or connecting
    if (connectionState === "connected" || connectionState === "connecting") {
      return
    }

    try {
      setConnectionState("connecting")
      const socket = new WebSocket(url)

      socket.onopen = (event) => {
        setConnectionState("connected")
        reconnectAttemptsRef.current = 0
        if (callbacks?.onOpen) callbacks.onOpen(event)
      }

      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          const message: WebSocketMessage = {
            id: Date.now().toString(),
            timestamp: new Date(),
            ...data,
          }

          if (!pausedRef.current) {
            setLastMessage(message)
            setMessageHistory((prev) => [...prev, message])
            if (callbacks?.onMessage) callbacks.onMessage(message)
          }
        } catch (error) {
          console.error("Error parsing WebSocket message:", error)
        }
      }

      socket.onclose = (event) => {
        setConnectionState("disconnected")
        if (callbacks?.onClose) callbacks.onClose(event)

        // Attempt to reconnect if not closed cleanly and we haven't exceeded max attempts
        if (!event.wasClean && reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current += 1
          if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current)
          reconnectTimeoutRef.current = setTimeout(() => {
            connect()
          }, reconnectInterval)
        }
      }

      socket.onerror = (event) => {
        if (callbacks?.onError) callbacks.onError(event)
      }

      socketRef.current = socket
    } catch (error) {
      console.error("Error connecting to WebSocket:", error)
      setConnectionState("disconnected")
    }
  }, [url, connectionState, callbacks, reconnectInterval, maxReconnectAttempts])

  const disconnect = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.close()
      socketRef.current = null
    }
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }
    setConnectionState("disconnected")
  }, [])

  const sendMessage = useCallback(
    (message: any) => {
      if (socketRef.current && connectionState === "connected") {
        socketRef.current.send(JSON.stringify(message))
        return true
      }
      return false
    },
    [connectionState],
  )

  const togglePause = useCallback(() => {
    setIsPaused((prev) => !prev)
  }, [])

  const clearMessages = useCallback(() => {
    setMessageHistory([])
    setLastMessage(null)
  }, [])

  // Connect on mount, disconnect on unmount
  useEffect(() => {
    connect()
    return () => {
      disconnect()
    }
  }, [connect, disconnect])

  return {
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
}
