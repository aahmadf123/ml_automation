"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

export function MLActivityFeed() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>ML Activity Feed</CardTitle>
        <CardDescription>Recent model-related activities and events</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="flex items-start space-x-4">
            <div className="relative mt-0.5">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-blue-100 text-blue-600">
                <span className="text-xs font-bold">JD</span>
              </div>
              <div className="absolute bottom-0 right-0 h-3 w-3 rounded-full border-2 border-background bg-green-500"></div>
            </div>
            <div className="flex-1 space-y-1">
              <div className="flex items-center justify-between">
                <div className="font-medium">John Doe deployed a new model version</div>
                <div className="text-xs text-muted-foreground">10m ago</div>
              </div>
              <div className="text-sm text-muted-foreground">Model v5 was deployed to production environment</div>
              <div className="flex items-center space-x-2">
                <Badge variant="outline">model-v5</Badge>
                <Badge variant="outline">production</Badge>
              </div>
            </div>
          </div>

          <div className="flex items-start space-x-4">
            <div className="relative mt-0.5">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-purple-100 text-purple-600">
                <span className="text-xs font-bold">AS</span>
              </div>
              <div className="absolute bottom-0 right-0 h-3 w-3 rounded-full border-2 border-background bg-green-500"></div>
            </div>
            <div className="flex-1 space-y-1">
              <div className="flex items-center justify-between">
                <div className="font-medium">Alice Smith updated feature importance analysis</div>
                <div className="text-xs text-muted-foreground">2h ago</div>
              </div>
              <div className="text-sm text-muted-foreground">Added 3 new features to the analysis dashboard</div>
              <div className="flex items-center space-x-2">
                <Badge variant="outline">feature-analysis</Badge>
                <Badge variant="outline">dashboard</Badge>
              </div>
            </div>
          </div>

          <div className="flex items-start space-x-4">
            <div className="relative mt-0.5">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-amber-100 text-amber-600">
                <span className="text-xs font-bold">RJ</span>
              </div>
              <div className="absolute bottom-0 right-0 h-3 w-3 rounded-full border-2 border-background bg-gray-500"></div>
            </div>
            <div className="flex-1 space-y-1">
              <div className="flex items-center justify-between">
                <div className="font-medium">Robert Johnson created a new experiment</div>
                <div className="text-xs text-muted-foreground">1d ago</div>
              </div>
              <div className="text-sm text-muted-foreground">Testing a new ensemble approach for claim prediction</div>
              <div className="flex items-center space-x-2">
                <Badge variant="outline">experiment</Badge>
                <Badge variant="outline">ensemble</Badge>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
