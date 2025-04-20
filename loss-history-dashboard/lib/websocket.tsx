import { createContext, useContext, useEffect, useState, ReactNode } from "react"

interface WebSocketMessage {
  type: string
  data: any
  timestamp: number
}

interface WebSocketContextType {
  connected: boolean
  lastMessage: WebSocketMessage | null
  sendMessage: (message: any) => void
  reconnect: () => void
}

const WebSocketContext = createContext<WebSocketContextType | null>(null)

export function WebSocketProvider({ children }: { children: ReactNode }) {
  const [socket, setSocket] = useState<WebSocket | null>(null)
  const [connected, setConnected] = useState(false)
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null)
  const [reconnectAttempt, setReconnectAttempt] = useState(0)

  const connect = () => {
    const ws = new WebSocket("ws://localhost:8000/ws/metrics")

    ws.onopen = () => {
      console.log("WebSocket connected")
      setConnected(true)
      setReconnectAttempt(0)
    }

    ws.onclose = () => {
      console.log("WebSocket disconnected")
      setConnected(false)
      // Attempt to reconnect with exponential backoff
      const timeout = Math.min(1000 * Math.pow(2, reconnectAttempt), 30000)
      setTimeout(() => {
        setReconnectAttempt(prev => prev + 1)
        connect()
      }, timeout)
    }

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data)
        setLastMessage({
          type: message.type,
          data: message.data,
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
  }

  useEffect(() => {
    connect()
    return () => {
      if (socket) {
        socket.close()
      }
    }
  }, [])

  const sendMessage = (message: any) => {
    if (socket && connected) {
      socket.send(JSON.stringify(message))
    }
  }

  const reconnect = () => {
    if (socket) {
      socket.close()
    }
    connect()
  }

  return (
    <WebSocketContext.Provider value={{ connected, lastMessage, sendMessage, reconnect }}>
      {children}
    </WebSocketContext.Provider>
  )
}

export function useWebSocket() {
  const context = useContext(WebSocketContext)
  if (!context) {
    throw new Error("useWebSocket must be used within a WebSocketProvider")
  }
  return context
} 