"use client"

import { DashboardHeader } from "@/components/dashboard-header"
import { PageHeader } from "@/components/page-header"
import { ThresholdManager } from "@/components/threshold-manager"
import { ThresholdVisualizer } from "@/components/threshold-visualizer"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { AlertTriangle, Bell, Info } from "lucide-react"

export default function ThresholdManagementPage() {
  return (
    <div className="flex flex-col min-h-screen">
      <DashboardHeader />
      <div className="flex-1 p-6 space-y-6">
        <PageHeader
          title="Threshold Management"
          description="Configure and manage custom alert thresholds for model drift and performance metrics"
        />

        <Tabs defaultValue="overview" className="space-y-4">
          <TabsList>
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="thresholds">Manage Thresholds</TabsTrigger>
            <TabsTrigger value="visualization">Visualization</TabsTrigger>
            <TabsTrigger value="alerts">Alert History</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg flex items-center">
                    <Bell className="mr-2 h-5 w-5 text-blue-500" />
                    Active Thresholds
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold">5</div>
                  <p className="text-sm text-muted-foreground">Across all metrics</p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg flex items-center">
                    <AlertTriangle className="mr-2 h-5 w-5 text-red-500" />
                    Current Violations
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold text-red-500">2</div>
                  <p className="text-sm text-muted-foreground">Requiring attention</p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg flex items-center">
                    <Info className="mr-2 h-5 w-5 text-yellow-500" />
                    Alert Frequency
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold">12</div>
                  <p className="text-sm text-muted-foreground">Alerts in the last 7 days</p>
                </CardContent>
              </Card>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <ThresholdVisualizer />

              <Card>
                <CardHeader>
                  <CardTitle>Recent Alerts</CardTitle>
                  <CardDescription>Threshold violations in the last 7 days</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="p-3 rounded-md border border-red-200 bg-red-50">
                      <div className="flex items-start space-x-3">
                        <AlertTriangle className="h-5 w-5 text-red-500 mt-0.5" />
                        <div>
                          <div className="flex items-center justify-between">
                            <h3 className="font-medium text-red-800">Property Value Drift Exceeded</h3>
                            <span className="text-xs text-red-700">2 hours ago</span>
                          </div>
                          <p className="mt-1 text-sm text-red-700">
                            Property Value Drift reached 5.5%, exceeding the threshold of 5%
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className="p-3 rounded-md border border-red-200 bg-red-50">
                      <div className="flex items-start space-x-3">
                        <AlertTriangle className="h-5 w-5 text-red-500 mt-0.5" />
                        <div>
                          <div className="flex items-center justify-between">
                            <h3 className="font-medium text-red-800">Location Risk Drift Critical</h3>
                            <span className="text-xs text-red-700">1 day ago</span>
                          </div>
                          <p className="mt-1 text-sm text-red-700">
                            Location Risk Drift reached 6.2%, exceeding the critical threshold of 5%
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className="p-3 rounded-md border border-yellow-200 bg-yellow-50">
                      <div className="flex items-start space-x-3">
                        <AlertTriangle className="h-5 w-5 text-yellow-500 mt-0.5" />
                        <div>
                          <div className="flex items-center justify-between">
                            <h3 className="font-medium text-yellow-800">RMSE Approaching Threshold</h3>
                            <span className="text-xs text-yellow-700">3 days ago</span>
                          </div>
                          <p className="mt-1 text-sm text-yellow-700">
                            RMSE reached 0.48, approaching the warning threshold of 0.5
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className="p-3 rounded-md border border-blue-200 bg-blue-50">
                      <div className="flex items-start space-x-3">
                        <Info className="h-5 w-5 text-blue-500 mt-0.5" />
                        <div>
                          <div className="flex items-center justify-between">
                            <h3 className="font-medium text-blue-800">R² Score Improved</h3>
                            <span className="text-xs text-blue-700">5 days ago</span>
                          </div>
                          <p className="mt-1 text-sm text-blue-700">
                            R² Score improved to 0.87, now above the minimum threshold of 0.8
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="thresholds" className="space-y-4">
            <ThresholdManager />
          </TabsContent>

          <TabsContent value="visualization" className="space-y-4">
            <ThresholdVisualizer />
          </TabsContent>

          <TabsContent value="alerts" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Alert History</CardTitle>
                <CardDescription>Historical record of threshold violations</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="p-3 rounded-md border">
                    <div className="flex items-start space-x-3">
                      <AlertTriangle className="h-5 w-5 text-red-500 mt-0.5" />
                      <div>
                        <div className="flex items-center justify-between">
                          <h3 className="font-medium">Property Value Drift Exceeded</h3>
                          <span className="text-xs text-muted-foreground">April 15, 2023 - 10:23 AM</span>
                        </div>
                        <p className="mt-1 text-sm text-muted-foreground">
                          Property Value Drift reached 5.5%, exceeding the threshold of 5%
                        </p>
                        <div className="mt-2 flex items-center text-xs text-muted-foreground">
                          <span className="bg-red-100 text-red-800 px-2 py-0.5 rounded-full">Critical</span>
                          <span className="mx-2">•</span>
                          <span>Notified: Email, Dashboard</span>
                          <span className="mx-2">•</span>
                          <span>Acknowledged by: John Doe</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="p-3 rounded-md border">
                    <div className="flex items-start space-x-3">
                      <AlertTriangle className="h-5 w-5 text-red-500 mt-0.5" />
                      <div>
                        <div className="flex items-center justify-between">
                          <h3 className="font-medium">Location Risk Drift Critical</h3>
                          <span className="text-xs text-muted-foreground">April 14, 2023 - 2:15 PM</span>
                        </div>
                        <p className="mt-1 text-sm text-muted-foreground">
                          Location Risk Drift reached 6.2%, exceeding the critical threshold of 5%
                        </p>
                        <div className="mt-2 flex items-center text-xs text-muted-foreground">
                          <span className="bg-red-100 text-red-800 px-2 py-0.5 rounded-full">Critical</span>
                          <span className="mx-2">•</span>
                          <span>Notified: Email, Slack, Dashboard</span>
                          <span className="mx-2">•</span>
                          <span>Acknowledged by: Jane Smith</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="p-3 rounded-md border">
                    <div className="flex items-start space-x-3">
                      <AlertTriangle className="h-5 w-5 text-yellow-500 mt-0.5" />
                      <div>
                        <div className="flex items-center justify-between">
                          <h3 className="font-medium">RMSE Approaching Threshold</h3>
                          <span className="text-xs text-muted-foreground">April 12, 2023 - 9:30 AM</span>
                        </div>
                        <p className="mt-1 text-sm text-muted-foreground">
                          RMSE reached 0.48, approaching the warning threshold of 0.5
                        </p>
                        <div className="mt-2 flex items-center text-xs text-muted-foreground">
                          <span className="bg-yellow-100 text-yellow-800 px-2 py-0.5 rounded-full">Warning</span>
                          <span className="mx-2">•</span>
                          <span>Notified: Dashboard</span>
                          <span className="mx-2">•</span>
                          <span>Auto-resolved</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="p-3 rounded-md border">
                    <div className="flex items-start space-x-3">
                      <Info className="h-5 w-5 text-blue-500 mt-0.5" />
                      <div>
                        <div className="flex items-center justify-between">
                          <h3 className="font-medium">R² Score Improved</h3>
                          <span className="text-xs text-muted-foreground">April 10, 2023 - 3:45 PM</span>
                        </div>
                        <p className="mt-1 text-sm text-muted-foreground">
                          R² Score improved to 0.87, now above the minimum threshold of 0.8
                        </p>
                        <div className="mt-2 flex items-center text-xs text-muted-foreground">
                          <span className="bg-blue-100 text-blue-800 px-2 py-0.5 rounded-full">Info</span>
                          <span className="mx-2">•</span>
                          <span>Notified: Dashboard</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
