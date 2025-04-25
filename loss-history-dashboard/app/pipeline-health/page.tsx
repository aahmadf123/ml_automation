"use client";

import { PipelineHealth } from "@/components/pipeline-health";
import DashboardHeader from "@/components/dashboard-header";
import DashboardSidebar from "@/components/dashboard-sidebar";

export default function PipelineHealthPage() {
  return (
    <div className="flex min-h-screen flex-col">
      <DashboardHeader title="Pipeline Health" />
      <div className="container flex-1 items-start md:grid md:grid-cols-[220px_minmax(0,1fr)] md:gap-6 lg:grid-cols-[240px_minmax(0,1fr)] lg:gap-10">
        <DashboardSidebar />
        <main className="flex w-full flex-col overflow-hidden p-4 md:p-6">
          <PipelineHealth />
        </main>
      </div>
    </div>
  );
} 