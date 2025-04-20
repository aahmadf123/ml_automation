import { DashboardHeader } from "@/components/dashboard-header"
import { PageHeader } from "@/components/page-header"
import { Skeleton } from "@/components/ui/skeleton"

export default function Loading() {
  return (
    <div className="flex flex-col min-h-screen">
      <DashboardHeader />
      <div className="flex-1 p-6 space-y-6">
        <PageHeader
          title="Model Explainability"
          description="Understand how your models make predictions and which features matter most"
        />

        <div className="space-y-4">
          <Skeleton className="h-10 w-full max-w-md" />

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Skeleton className="h-[500px] w-full" />
            <Skeleton className="h-[500px] w-full" />
          </div>

          <Skeleton className="h-[400px] w-full" />
        </div>
      </div>
    </div>
  )
}
