"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { MiniSparkline } from "@/components/mini-sparkline"
import { Activity, CheckCircle, Clock, AlertTriangle } from "lucide-react"

export function QuickStats() {
  // Sample data for the sparklines
  const accuracyData = [92.1, 92.4, 93.0, 93.2, 93.8, 94.2]
  const claimsData = [1150, 1180, 1210, 1240, 1260, 1284]
  const timeData = [4.1, 3.9, 3.7, 3.5, 3.3, 3.2]
  const anomalyData = [31, 29, 27, 26, 25, 24]

  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Prediction Accuracy</CardTitle>
          <Activity className="h-4 w-4 text-emerald-500" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">94.2%</div>
          <div className="flex items-center text-xs text-muted-foreground">
            <span className="flex items-center text-emerald-500">
              <span className="mr-1">↑</span> 2.1%
            </span>
            <MiniSparkline data={accuracyData} color="#10b981" className="ml-2 h-8" />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Claims Processed</CardTitle>
          <CheckCircle className="h-4 w-4 text-emerald-500" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">1,284</div>
          <div className="flex items-center text-xs text-muted-foreground">
            <span className="flex items-center text-emerald-500">
              <span className="mr-1">↑</span> 12.5%
            </span>
            <MiniSparkline data={claimsData} color="#10b981" className="ml-2 h-8" />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Avg. Processing Time</CardTitle>
          <Clock className="h-4 w-4 text-rose-500" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">3.2 days</div>
          <div className="flex items-center text-xs text-muted-foreground">
            <span className="flex items-center text-rose-500">
              <span className="mr-1">↓</span> 8.4%
            </span>
            <MiniSparkline data={timeData} color="#f43f5e" className="ml-2 h-8" inverted />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Anomalies Detected</CardTitle>
          <AlertTriangle className="h-4 w-4 text-rose-500" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">24</div>
          <div className="flex items-center text-xs text-muted-foreground">
            <span className="flex items-center text-rose-500">
              <span className="mr-1">↓</span> 15.3%
            </span>
            <MiniSparkline data={anomalyData} color="#f43f5e" className="ml-2 h-8" inverted />
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
