"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

export function FeatureDriftSnapshot() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Feature Drift Snapshot</CardTitle>
        <CardDescription>Top drifting features in the last 7 days</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <div className="font-medium">ClaimAmount</div>
              <Badge variant="destructive">High Drift</Badge>
            </div>
            <div className="text-sm font-medium">PSI: 0.28</div>
          </div>
          <div className="h-2 w-full rounded-full bg-muted">
            <div className="h-2 rounded-full bg-red-500" style={{ width: "85%" }}></div>
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <div className="font-medium">PropertyLocation</div>
              <Badge variant="warning">Medium Drift</Badge>
            </div>
            <div className="text-sm font-medium">PSI: 0.15</div>
          </div>
          <div className="h-2 w-full rounded-full bg-muted">
            <div className="h-2 rounded-full bg-amber-500" style={{ width: "60%" }}></div>
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <div className="font-medium">InsuredAge</div>
              <Badge variant="warning">Medium Drift</Badge>
            </div>
            <div className="text-sm font-medium">PSI: 0.12</div>
          </div>
          <div className="h-2 w-full rounded-full bg-muted">
            <div className="h-2 rounded-full bg-amber-500" style={{ width: "48%" }}></div>
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <div className="font-medium">ClaimType</div>
              <Badge variant="outline">Low Drift</Badge>
            </div>
            <div className="text-sm font-medium">PSI: 0.05</div>
          </div>
          <div className="h-2 w-full rounded-full bg-muted">
            <div className="h-2 rounded-full bg-blue-500" style={{ width: "20%" }}></div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
