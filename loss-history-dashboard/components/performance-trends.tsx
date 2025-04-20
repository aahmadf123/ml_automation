"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

export function PerformanceTrends() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Performance Trends</CardTitle>
        <CardDescription>Model metrics over the last 30 days</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="accuracy">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="accuracy">Accuracy</TabsTrigger>
            <TabsTrigger value="precision">Precision</TabsTrigger>
            <TabsTrigger value="recall">Recall</TabsTrigger>
            <TabsTrigger value="f1">F1 Score</TabsTrigger>
          </TabsList>
          <TabsContent value="accuracy" className="h-[200px] w-full">
            <div className="flex h-full items-center justify-center">
              <div className="text-sm text-muted-foreground">Accuracy trend chart will appear here</div>
            </div>
          </TabsContent>
          <TabsContent value="precision" className="h-[200px] w-full">
            <div className="flex h-full items-center justify-center">
              <div className="text-sm text-muted-foreground">Precision trend chart will appear here</div>
            </div>
          </TabsContent>
          <TabsContent value="recall" className="h-[200px] w-full">
            <div className="flex h-full items-center justify-center">
              <div className="text-sm text-muted-foreground">Recall trend chart will appear here</div>
            </div>
          </TabsContent>
          <TabsContent value="f1" className="h-[200px] w-full">
            <div className="flex h-full items-center justify-center">
              <div className="text-sm text-muted-foreground">F1 Score trend chart will appear here</div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
