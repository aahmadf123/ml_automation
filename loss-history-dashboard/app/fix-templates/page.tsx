"use client"

import { DashboardHeader } from "@/components/dashboard-header"
import { PageHeader } from "@/components/page-header"
import { FixTemplatesLibrary } from "@/components/fix-templates-library"

export default function FixTemplatesPage() {
  return (
    <div className="flex flex-col min-h-screen">
      <DashboardHeader />
      <div className="flex-1 p-6 space-y-6">
        <PageHeader title="Fix Templates" description="Manage and apply templates for automated fixes" />

        <FixTemplatesLibrary />
      </div>
    </div>
  )
}
