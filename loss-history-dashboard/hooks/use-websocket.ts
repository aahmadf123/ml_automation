"use client"

import { useEffect, useRef, useState } from 'react'

interface WebSocketMessage {
  data: string
}

export function useWebSocket(url: string) {
  const [isConnected, setIsConnected] = useState(false)
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null)
  const ws = useRef<WebSocket | null>(null)
  const reconnectTimeout = useRef<NodeJS.Timeout>()

  useEffect(() => {
    const connect = () => {
      try {
        ws.current = new WebSocket(url)

        ws.current.onopen = () => {
          setIsConnected(true)
          console.log('WebSocket connected')
        }

        ws.current.onclose = () => {
          setIsConnected(false)
          console.log('WebSocket disconnected')
          // Attempt to reconnect after 5 seconds
          reconnectTimeout.current = setTimeout(connect, 5000)
        }

        ws.current.onerror = (error) => {
          console.error('WebSocket error:', error)
        }

        ws.current.onmessage = (event) => {
          setLastMessage(event)
        }
      } catch (error) {
        console.error('WebSocket connection error:', error)
        // Attempt to reconnect after 5 seconds
        reconnectTimeout.current = setTimeout(connect, 5000)
      }
    }

    connect()

    return () => {
      if (ws.current) {
        ws.current.close()
      }
      if (reconnectTimeout.current) {
        clearTimeout(reconnectTimeout.current)
      }
    }
  }, [url])

  const sendMessage = (message: string) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(message)
    } else {
      console.warn('WebSocket is not connected')
    }
  }

  return {
    isConnected,
    lastMessage,
    sendMessage
  }
}
