"use client"

import { type ReactNode, createContext, useContext } from "react"

// Create a simple mock WebSocketContext that provides the expected interface
// but doesn't actually connect to any WebSocket
interface WebSocketContextType {
  connectionState: "connected" | "connecting" | "disconnected" | "error"
  lastMessage: any
  messageHistory: any[]
  isPaused: boolean
  connect: () => void
  disconnect: () => void
  sendMessage: (message: any) => boolean
  togglePause: () => void
  clearMessages: () => void
}

const defaultContextValue: WebSocketContextType = {
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

const WebSocketContext = createContext<WebSocketContextType>(defaultContextValue)

export function useWebSocketContext() {
  return useContext(WebSocketContext)
}

interface WebSocketWrapperProps {
  children: ReactNode
}

export function WebSocketWrapper({ children }: WebSocketWrapperProps) {
  // Use the default context value
  return <WebSocketContext.Provider value={defaultContextValue}>{children}</WebSocketContext.Provider>
}
