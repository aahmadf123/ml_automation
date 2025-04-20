"use client"

import { DashboardHeader } from "@/components/dashboard-header"
import { PageHeader } from "@/components/page-header"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { DataVisualization } from "@/components/data-visualization"

// Sample data for visualization
const sampleData = [
  { month: "Jan", sales: 1000, profit: 500, customers: 100, category: "Electronics" },
  { month: "Feb", sales: 1500, profit: 700, customers: 150, category: "Electronics" },
  { month: "Mar", sales: 1200, profit: 600, customers: 120, category: "Electronics" },
  { month: "Apr", sales: 1800, profit: 900, customers: 180, category: "Clothing" },
  { month: "May", sales: 2000, profit: 1000, customers: 200, category: "Clothing" },
  { month: "Jun", sales: 2200, profit: 1100, customers: 220, category: "Clothing" },
  { month: "Jul", sales: 1900, profit: 950, customers: 190, category: "Home" },
  { month: "Aug", sales: 2100, profit: 1050, customers: 210, category: "Home" },
  { month: "Sep", sales: 2300, profit: 1150, customers: 230, category: "Electronics" },
  { month: "Oct", sales: 2500, profit: 1250, customers: 250, category: "Electronics" },
  { month: "Nov", sales: 2700, profit: 1350, customers: 270, category: "Home" },
  { month: "Dec", sales: 3000, profit: 1500, customers: 300, category: "Clothing" },
]

export default function VisualizationsPage() {
  return (
    <div className="flex flex-col min-h-screen">
      <DashboardHeader />
      <div className="flex-1 p-6 space-y-6">
        <PageHeader
          title="Visualizations"
          description="Explore and visualize your data with interactive charts and graphs"
        />

        <Tabs defaultValue="overview" className="space-y-4">
          <TabsList>
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="trends">Trends</TabsTrigger>
            <TabsTrigger value="distributions">Distributions</TabsTrigger>
            <TabsTrigger value="correlations">Correlations</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-4">
            <DataVisualization data={sampleData} />
          </TabsContent>

          <TabsContent value="trends" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Trend Analysis</CardTitle>
                <CardDescription>Analyze trends in your data over time</CardDescription>
              </CardHeader>
              <CardContent className="h-[400px] flex items-center justify-center">
                <p className="text-muted-foreground">Trend visualization will appear here</p>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="distributions" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Data Distributions</CardTitle>
                <CardDescription>Visualize the distribution of your data</CardDescription>
              </CardHeader>
              <CardContent className="h-[400px] flex items-center justify-center">
                <p className="text-muted-foreground">Distribution visualization will appear here</p>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="correlations" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Correlation Analysis</CardTitle>
                <CardDescription>Explore correlations between different variables</CardDescription>
              </CardHeader>
              <CardContent className="h-[400px] flex items-center justify-center">
                <p className="text-muted-foreground">Correlation visualization will appear here</p>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
