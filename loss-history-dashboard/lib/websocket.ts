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
