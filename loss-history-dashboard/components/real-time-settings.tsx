"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Button } from "@/components/ui/button"
import { useWebSocketContext } from "@/components/websocket-provider"
import { RefreshCw, Zap } from "lucide-react"

export function RealTimeSettings() {
  const { connectionState, isPaused, togglePause, connect, disconnect } = useWebSocketContext()
  const [notificationsEnabled, setNotificationsEnabled] = useState(true)
  const [updateFrequency, setUpdateFrequency] = useState([5])
  const [autoReconnect, setAutoReconnect] = useState(true)
  const [showVisualIndicators, setShowVisualIndicators] = useState(true)

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">Real-Time Settings</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="flex items-center justify-between">
          <div className="space-y-0.5">
            <Label htmlFor="real-time-updates">Real-Time Updates</Label>
            <p className="text-sm text-muted-foreground">Enable live data updates</p>
          </div>
          <Switch
            id="real-time-updates"
            checked={connectionState === "open"}
            onCheckedChange={(checked) => {
              if (checked) {
                connect()
              } else {
                disconnect()
              }
            }}
          />
        </div>

        <div className="flex items-center justify-between">
          <div className="space-y-0.5">
            <Label htmlFor="pause-updates">Pause Updates</Label>
            <p className="text-sm text-muted-foreground">Temporarily pause incoming updates</p>
          </div>
          <Switch
            id="pause-updates"
            checked={!isPaused}
            onCheckedChange={() => togglePause()}
            disabled={connectionState !== "open"}
          />
        </div>

        <div className="flex items-center justify-between">
          <div className="space-y-0.5">
            <Label htmlFor="notifications">Notifications</Label>
            <p className="text-sm text-muted-foreground">Show toast notifications for important events</p>
          </div>
          <Switch id="notifications" checked={notificationsEnabled} onCheckedChange={setNotificationsEnabled} />
        </div>

        <div className="flex items-center justify-between">
          <div className="space-y-0.5">
            <Label htmlFor="visual-indicators">Visual Indicators</Label>
            <p className="text-sm text-muted-foreground">Show visual highlights for updated components</p>
          </div>
          <Switch id="visual-indicators" checked={showVisualIndicators} onCheckedChange={setShowVisualIndicators} />
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label htmlFor="update-frequency">Update Frequency</Label>
            <span className="text-sm">{updateFrequency[0]} seconds</span>
          </div>
          <Slider
            id="update-frequency"
            value={updateFrequency}
            onValueChange={setUpdateFrequency}
            min={1}
            max={30}
            step={1}
            disabled={connectionState !== "open"}
          />
        </div>

        <div className="flex items-center justify-between">
          <div className="space-y-0.5">
            <Label htmlFor="auto-reconnect">Auto Reconnect</Label>
            <p className="text-sm text-muted-foreground">Automatically reconnect if connection is lost</p>
          </div>
          <Switch id="auto-reconnect" checked={autoReconnect} onCheckedChange={setAutoReconnect} />
        </div>

        <div className="flex justify-end space-x-2">
          <Button variant="outline" onClick={() => connect()} disabled={connectionState === "open"}>
            <RefreshCw className="mr-2 h-4 w-4" />
            Reconnect
          </Button>
          <Button
            onClick={() => {
              // Reset all settings to default
              setNotificationsEnabled(true)
              setUpdateFrequency([5])
              setAutoReconnect(true)
              setShowVisualIndicators(true)
              if (connectionState !== "open") {
                connect()
              }
            }}
          >
            <Zap className="mr-2 h-4 w-4" />
            Apply Settings
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
