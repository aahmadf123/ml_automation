"use client"

import { usePathname } from "next/navigation"
import { useMemo } from "react"

export type Breadcrumb = {
  label: string
  href: string
  active: boolean
}

const routeLabels: Record<string, string> = {
  "": "Home",
  "data-ingestion": "Data Ingestion",
  "data-quality": "Data Quality",
  visualizations: "Visualizations",
  "model-metrics": "Model Metrics",
  "drift-monitor": "Drift Monitor",
  incidents: "Incidents",
  "code-interpreter": "Code Interpreter",
  "fix-templates": "Fix Templates",
  "notification-analytics": "Notification Analytics",
  settings: "Settings",
}

export function useBreadcrumbs(): Breadcrumb[] {
  const pathname = usePathname()

  const breadcrumbs = useMemo(() => {
    // Remove trailing slash and split the path
    const segments = pathname.replace(/\/$/, "").split("/").filter(Boolean)

    // Always start with home
    const crumbs: Breadcrumb[] = [
      {
        label: "Home",
        href: "/",
        active: segments.length === 0,
      },
    ]

    // Build up the breadcrumbs based on path segments
    let path = ""
    segments.forEach((segment, index) => {
      path += `/${segment}`
      crumbs.push({
        label: routeLabels[segment] || segment.charAt(0).toUpperCase() + segment.slice(1).replace(/-/g, " "),
        href: path,
        active: index === segments.length - 1,
      })
    })

    return crumbs
  }, [pathname])

  return breadcrumbs
}
