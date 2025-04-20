"use client"

import { DashboardHeader } from "@/components/dashboard-header"
import { PageHeader } from "@/components/page-header"
import { CodeInterpreter } from "@/components/code-interpreter"

export default function CodeInterpreterPage() {
  return (
    <div className="flex flex-col min-h-screen">
      <DashboardHeader />
      <div className="flex-1 p-6 space-y-6">
        <PageHeader title="Code Interpreter" description="Execute and analyze code for data processing and analysis" />

        <div className="grid gap-4">
          <CodeInterpreter />
        </div>
      </div>
    </div>
  )
}
