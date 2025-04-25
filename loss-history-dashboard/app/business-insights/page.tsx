"use client"

import { DashboardHeader } from "@/components/dashboard-header"
import { DashboardSidebar } from "@/components/dashboard-sidebar"
import BusinessInsights from "@/components/business-insights"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ArrowDown, ArrowUp, DollarSign, PieChart, TrendingUp, Users } from "lucide-react"

export default function BusinessInsightsPage() {
  return (
    <div className="flex min-h-screen flex-col">
      <DashboardHeader title="Business Insights" />
      <div className="container flex-1 items-start md:grid md:grid-cols-[220px_minmax(0,1fr)] md:gap-6 lg:grid-cols-[240px_minmax(0,1fr)] lg:gap-10">
        <DashboardSidebar />
        <main className="flex w-full flex-col overflow-hidden p-4 md:p-6">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4 mb-8">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">
                  Estimated Annual Savings
                </CardTitle>
                <DollarSign className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">$2.4M</div>
                <p className="text-xs text-muted-foreground">
                  <span className="text-green-500 flex items-center gap-1">
                    <ArrowUp className="h-3 w-3" /> 18.2%
                  </span>{" "}
                  from last quarter
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">
                  Loss Ratio Improvement
                </CardTitle>
                <TrendingUp className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">3.4%</div>
                <p className="text-xs text-muted-foreground">
                  <span className="text-green-500 flex items-center gap-1">
                    <ArrowUp className="h-3 w-3" /> 0.8%
                  </span>{" "}
                  from last quarter
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">
                  Pricing Accuracy
                </CardTitle>
                <PieChart className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">94.7%</div>
                <p className="text-xs text-muted-foreground">
                  <span className="text-green-500 flex items-center gap-1">
                    <ArrowUp className="h-3 w-3" /> 2.3%
                  </span>{" "}
                  from last quarter
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">
                  Customer Retention
                </CardTitle>
                <Users className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">89.2%</div>
                <p className="text-xs text-muted-foreground">
                  <span className="text-red-500 flex items-center gap-1">
                    <ArrowDown className="h-3 w-3" /> 1.1%
                  </span>{" "}
                  from last quarter
                </p>
              </CardContent>
            </Card>
          </div>
          
          <BusinessInsights />
        </main>
      </div>
    </div>
  )
} 
