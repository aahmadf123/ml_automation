"use client"

import { DashboardHeader } from "@/components/dashboard-header"
import { PageHeader } from "@/components/page-header"
import { RecentClaims } from "@/components/recent-claims"

export default function ClaimsPage() {
  return (
    <div className="flex flex-col min-h-screen">
      <DashboardHeader />
      <div className="flex-1 p-6 space-y-6 md:ml-64">
        <PageHeader title="Claims" description="Claims processing and analysis" />

        <RecentClaims />

        {/* Additional claims-related components can be added here */}
      </div>
    </div>
  )
}
