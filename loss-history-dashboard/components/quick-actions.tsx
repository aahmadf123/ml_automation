"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { RefreshCw, Zap, Lightbulb, Terminal } from "lucide-react"
import { useState } from "react"

export function QuickActions() {
  const [isConsoleOpen, setIsConsoleOpen] = useState(false)

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">Quick Actions</CardTitle>
      </CardHeader>
      <CardContent className="grid grid-cols-2 gap-4">
        <Button variant="outline" className="h-20 flex flex-col items-center justify-center space-y-1 rounded-xl">
          <RefreshCw className="h-5 w-5" />
          <span className="text-xs">Trigger Retrain</span>
        </Button>
        <Button variant="outline" className="h-20 flex flex-col items-center justify-center space-y-1 rounded-xl">
          <Zap className="h-5 w-5" />
          <span className="text-xs">Run Self-Heal</span>
        </Button>
        <Button variant="outline" className="h-20 flex flex-col items-center justify-center space-y-1 rounded-xl">
          <Lightbulb className="h-5 w-5" />
          <span className="text-xs">Generate Fix</span>
        </Button>
        <Button
          variant="outline"
          className="h-20 flex flex-col items-center justify-center space-y-1 rounded-xl"
          onClick={() => setIsConsoleOpen(true)}
        >
          <Terminal className="h-5 w-5" />
          <span className="text-xs">Code Console</span>
        </Button>
      </CardContent>
    </Card>
  )
}
