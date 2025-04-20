"use client"

import { useState } from "react"
import { DashboardHeader } from "@/components/dashboard-header"
import { PageHeader } from "@/components/page-header"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { NotificationSummaryCards } from "@/components/notification-summary-cards"
import { NotificationDistributionChart } from "@/components/notification-distribution-chart"
import { NotificationTrendsChart } from "@/components/notification-trends-chart"
import { ResolutionTimeChart } from "@/components/resolution-time-chart"
import { NotificationTable } from "@/components/notification-table"
import { NotificationDashboardFilters } from "@/components/notification-dashboard-filters"
import { RealTimeNotificationFeed } from "@/components/real-time-notification-feed"
import { RealTimeNotificationMetrics } from "@/components/real-time-notification-metrics"
import { RealTimeNotificationChart } from "@/components/real-time-notification-chart"
import { RealTimeNotificationStatus } from "@/components/real-time-notification-status"

export default function NotificationAnalyticsPage() {
  const [dateRange, setDateRange] = useState<[Date | undefined, Date | undefined]>([
    new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
    new Date(),
  ])

  return (
    <div className="flex flex-col min-h-screen">
      <DashboardHeader />
      <div className="flex-1 p-6 space-y-6">
        <PageHeader
          title="Notification Analytics"
          description="Monitor and analyze notification metrics by priority level"
        >
          <NotificationDashboardFilters dateRange={dateRange} setDateRange={setDateRange} />
        </PageHeader>

        <Tabs defaultValue="overview" className="space-y-4">
          <TabsList>
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="real-time">Real-Time</TabsTrigger>
            <TabsTrigger value="details">Details</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-4">
            <NotificationSummaryCards />

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <NotificationDistributionChart />
              <NotificationTrendsChart dateRange={dateRange} />
            </div>

            <ResolutionTimeChart />

            <NotificationTable />
          </TabsContent>

          <TabsContent value="real-time" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <RealTimeNotificationMetrics />
              <RealTimeNotificationStatus />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <RealTimeNotificationChart className="col-span-2" />
              <RealTimeNotificationFeed />
            </div>
          </TabsContent>

          <TabsContent value="details" className="space-y-4">
            <NotificationTable />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
