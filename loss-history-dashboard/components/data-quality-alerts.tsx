"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { AlertTriangle, AlertCircle, Info } from "lucide-react"

export function DataQualityAlerts() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Data Quality Alerts</CardTitle>
        <CardDescription>Recent data quality issues detected</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="rounded-md border p-4">
            <div className="flex items-start space-x-3">
              <AlertTriangle className="mt-0.5 h-5 w-5 text-amber-500" />
              <div>
                <div className="flex items-center space-x-2">
                  <div className="font-medium">Missing Values in ClaimAmount</div>
                  <Badge variant="warning">Medium</Badge>
                </div>
                <div className="mt-1 text-sm text-muted-foreground">
                  5.2% of records have missing ClaimAmount values in the latest batch
                </div>
                <div className="mt-2 text-xs text-muted-foreground">Detected 2 hours ago</div>
              </div>
            </div>
          </div>

          <div className="rounded-md border p-4">
            <div className="flex items-start space-x-3">
              <AlertCircle className="mt-0.5 h-5 w-5 text-red-500" />
              <div>
                <div className="flex items-center space-x-2">
                  <div className="font-medium">Outliers in PropertyValue</div>
                  <Badge variant="destructive">High</Badge>
                </div>
                <div className="mt-1 text-sm text-muted-foreground">
                  3 extreme outliers detected in PropertyValue column
                </div>
                <div className="mt-2 text-xs text-muted-foreground">Detected 5 hours ago</div>
              </div>
            </div>
          </div>

          <div className="rounded-md border p-4">
            <div className="flex items-start space-x-3">
              <Info className="mt-0.5 h-5 w-5 text-blue-500" />
              <div>
                <div className="flex items-center space-x-2">
                  <div className="font-medium">Data Type Mismatch</div>
                  <Badge variant="outline">Low</Badge>
                </div>
                <div className="mt-1 text-sm text-muted-foreground">Inconsistent date formats in ClaimDate column</div>
                <div className="mt-2 text-xs text-muted-foreground">Detected 1 day ago</div>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
