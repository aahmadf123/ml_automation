"use client"

import { DashboardHeader } from "@/components/dashboard-header"
import { DashboardSidebar } from "@/components/dashboard-sidebar"
import BusinessInsights from "@/components/business-insights"
import { DollarSign, PieChart, TrendingUp, Users } from "lucide-react"
import { SummaryCard } from "@/components/ui/summary-card"
import { ROIVisualization } from "@/components/roi-visualization"
import { CompetitiveAdvantageMetrics } from "@/components/competitive-advantage-metrics"
export default function BusinessInsightsPage() {
  return (
    <div className="flex min-h-screen flex-col">
      <DashboardHeader title="Business Insights" />
      <div className="container flex-1 items-start md:grid md:grid-cols-[220px_minmax(0,1fr)] md:gap-6 lg:grid-cols-[240px_minmax(0,1fr)] lg:gap-10">
        <DashboardSidebar />
        <main className="flex w-full flex-col overflow-hidden p-4 md:p-6">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4 mb-8">
            <SummaryCard
              title="Total Projected Savings"
              value="$2.4M"
              icon={DollarSign}
              trend="positive"
              changeValue="18.2%"
              tooltipText="Annual cost savings achieved through ML predictions compared to traditional methods."
              changeText="from last quarter"
              suffix=" annually"
            />
            <SummaryCard
              title="Loss Ratio Improvement"
              value="3.4"
              suffix="%"
              icon={TrendingUp}
              trend="positive"
              changeValue="0.8%"
              tooltipText="Reduction in the loss ratio through better predictive modeling and risk assessment."
            />
            <SummaryCard
              title="Pricing Accuracy"
              value="94.7"
              suffix="%"
              icon={PieChart}
              trend="positive"
              changeValue="2.3%"
              tooltipText="Accuracy of premium pricing models compared to actual losses."
            />
            <SummaryCard
              title="Customer Retention"
              value="89.2"
              suffix="%"
              icon={Users}
              trend="negative"
              changeValue="1.1%"
              tooltipText="Percentage of customers retained year-over-year after implementing ML-based pricing."
            />
          </div>
          
          </div>
          
          <BusinessInsights />
        </main>
      </div>
    </div>
  )
} 
