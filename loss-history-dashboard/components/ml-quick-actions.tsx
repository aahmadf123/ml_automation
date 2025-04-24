"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { RefreshCw, Play, RotateCcw, Download, AlertTriangle } from "lucide-react"

export function MLQuickActions() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>ML Quick Actions</CardTitle>
        <CardDescription>Common model operations and actions</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4">
          <Button className="flex items-center justify-start space-x-2">
            <Play className="h-4 w-4" />
            <span>Retrain Model</span>
          </Button>
          
          <Button variant="outline" className="flex items-center justify-start space-x-2">
            <RefreshCw className="h-4 w-4" />
            <span>Update Features</span>
          </Button>
          
          <Button variant="outline" className="flex items-center justify-start space-x-2">
            <RotateCcw className="h-4 w-4" />
            <span>Rollback Model</span>
          </Button>
          
          <Button variant="outline" className="flex items-center justify-start space-x-2">
            <Download className="h-4 w-4" />
            <span>Export Model</span>
          </Button>
          
          <Button variant="destructive" className="flex items-center justify-start space-x-2 col-span-2">
            <AlertTriangle className="h-4 w-4" />
            <span>Report Issue</span>
          </Button>
        </div>
        
        <div className="mt-4 rounded-md border p-4">
          <div className="text-sm font-medium">Recent Actions</div>
          <div className="mt-2 space-y-2 text-sm">
            <div className="flex items-center justify-between">
              <div>Model v4 deployed to production</div>
              <div className="text-xs text-muted-foreground">2 days ago</div>
            </div>
            <div className="flex items-center justify-between">
              <div>Feature importance analysis run</div>
              <div className="text-xs text-muted-foreground">3 days ago</div>
            </div>
            <div className="flex items-center justify-between">
              <div>Drift detection threshold updated</div>
              <div className="text-xs text-muted-foreground">5 days ago</div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
