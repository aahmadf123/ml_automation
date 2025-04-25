"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import {
  BarChart3,
  Bell,
  Code,
  Database,
  FileWarning,
  Gauge,
  Home,
  LineChart,
  Settings,
  AlertTriangle,
  Lightbulb,
  GitCompare,
  Workflow,
  ChevronDown,
  PieChart,
  TrendingUp,
  Activity,
  FileText,
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { useState } from "react"
import { cn } from "@/lib/utils"

interface NavItem {
  title: string
  href: string
  icon: any
  active: boolean
  description?: string
}

interface NavGroup {
  title: string
  items: NavItem[]
}

export function DashboardSidebar() {
  const pathname = usePathname()
  const [expandedGroups, setExpandedGroups] = useState<string[]>(["monitoring", "models", "analytics"])

  const isActive = (path: string) => {
    return pathname === path
  }

  const toggleGroup = (groupTitle: string) => {
    setExpandedGroups(prev =>
      prev.includes(groupTitle)
        ? prev.filter(g => g !== groupTitle)
        : [...prev, groupTitle]
    )
  }

  const navGroups: NavGroup[] = [
    {
      title: "Overview",
      items: [
        {
          title: "Dashboard",
          href: "/",
          icon: Home,
          active: isActive("/"),
          description: "Overall system performance and key metrics",
        },
      ],
    },
    {
      title: "Business Intelligence",
      items: [
        {
          title: "Loss Predictions",
          href: "/predictions",
          icon: TrendingUp,
          active: isActive("/predictions"),
          description: "Forecast potential losses with 95% confidence intervals",
        },
        {
          title: "Premium Projections",
          href: "/projections",
          icon: LineChart,
          active: isActive("/projections"),
          description: "Project future premium growth with risk adjustments",
        },
        {
          title: "Business Insights",
          href: "/business-insights",
          icon: PieChart,
          active: isActive("/business-insights"),
          description: "Key business metrics and ROI analysis",
        },
        {
          title: "Visualizations",
          href: "/visualizations",
          icon: BarChart3,
          active: isActive("/visualizations"),
          description: "Interactive data visualizations and reports",
        },
      ],
    },
    {
      title: "Claims Management",
      items: [
        {
          title: "Claims Analysis",
          href: "/claims",
          icon: FileText,
          active: isActive("/claims"),
          description: "Detailed claims analysis with cost breakdowns",
        },
        {
          title: "Notification Analytics",
          href: "/notification-analytics",
          icon: Bell,
          active: isActive("/notification-analytics"),
          description: "Track notification effectiveness and response rates",
        },
        {
          title: "Threshold Management",
          href: "/threshold-management",
          icon: Gauge,
          active: isActive("/threshold-management"),
          description: "Set and manage alert thresholds for claims",
        },
      ],
    },
    {
      title: "Monitoring",
      items: [
        {
          title: "Pipeline Health",
          href: "/pipeline-health",
          icon: Activity,
          active: isActive("/pipeline-health"),
          description: "Monitor data pipeline health and performance",
        },
        {
          title: "Drift Monitor",
          href: "/drift-monitor",
          icon: AlertTriangle,
          active: isActive("/drift-monitor"),
          description: "Track model drift and data distribution changes",
        },
        {
          title: "Data Quality",
          href: "/data-quality",
          icon: FileWarning,
          active: isActive("/data-quality"),
          description: "Monitor data quality metrics and anomalies",
        },
        {
          title: "Incidents",
          href: "/incidents",
          icon: Bell,
          active: isActive("/incidents"),
          description: "Track and manage system incidents",
        },
      ],
    },
    {
      title: "Model Hub",
      items: [
        {
          title: "Model Metrics",
          href: "/model-metrics",
          icon: Gauge,
          active: isActive("/model-metrics"),
          description: "Performance metrics across all models",
        },
        {
          title: "Model Comparison",
          href: "/model-comparison",
          icon: GitCompare,
          active: isActive("/model-comparison"),
          description: "Compare models side-by-side with business impact",
        },
        {
          title: "Model Explainability",
          href: "/model-explainability",
          icon: Lightbulb,
          active: isActive("/model-explainability"),
          description: "Understand what drives model predictions",
        },
        {
          title: "Explanation Comparison",
          href: "/model-explanation-comparison",
          icon: GitCompare,
          active: isActive("/model-explanation-comparison"),
          description: "Compare feature importance across models",
        },
      ],
    },
    {
      title: "Data Operations",
      items: [
        {
          title: "Data Ingestion",
          href: "/data-ingestion",
          icon: Database,
          active: isActive("/data-ingestion"),
          description: "Monitor and manage data ingestion pipelines",
        },
        {
          title: "Fix Proposals",
          href: "/fix-proposals",
          icon: Workflow,
          active: isActive("/fix-proposals"),
          description: "Review and implement proposed fixes",
        },
        {
          title: "Fix Templates",
          href: "/fix-templates",
          icon: Workflow,
          active: isActive("/fix-templates"),
          description: "Manage templates for automated fixes",
        },
      ],
    },
    {
      title: "Integration & Settings",
      items: [
        {
          title: "Integrations",
          href: "/integrations",
          icon: Code,
          active: isActive("/integrations"),
          description: "Manage connections with external systems",
        },
        {
          title: "Code Interpreter",
          href: "/code-interpreter",
          icon: Code,
          active: isActive("/code-interpreter"),
          description: "Run custom code for data analysis",
        },
        {
          title: "Settings",
          href: "/settings",
          icon: Settings,
          active: isActive("/settings"),
          description: "Configure system preferences and user settings",
        },
      ],
    },
  ]

  return (
    <div className="fixed inset-y-0 left-0 z-50 hidden w-64 border-r bg-background lg:block">
      <div className="flex h-full flex-col">
        <div className="flex h-14 items-center border-b px-4">
          <Link href="/" className="flex items-center gap-2">
            <span className="text-lg font-bold">Loss History Agent</span>
            <Badge variant="outline" className="bg-green-500 text-white border-green-500">
              Live
            </Badge>
          </Link>
        </div>
        <div className="flex-1 overflow-auto py-2">
          <nav className="grid items-start px-2 text-sm font-medium">
            {navGroups.map((group) => (
              <div key={group.title} className="space-y-1">
                <button
                  onClick={() => toggleGroup(group.title.toLowerCase())}
                  className={cn(
                    "flex w-full items-center justify-between rounded-lg px-3 py-2 text-gray-500 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-50",
                    expandedGroups.includes(group.title.toLowerCase()) && "bg-gray-100 dark:bg-gray-800"
                  )}
                >
                  <span>{group.title}</span>
                  <ChevronDown
                    className={cn(
                      "h-4 w-4 transition-transform",
                      expandedGroups.includes(group.title.toLowerCase()) && "rotate-180"
                    )}
                  />
                </button>
                {expandedGroups.includes(group.title.toLowerCase()) && (
                  <div className="space-y-1 pl-4">
                    {group.items.map((item, index) => (
                      <Link
                        key={index}
                        href={item.href}
                        className={cn(
                          "flex items-center gap-3 rounded-lg px-3 py-2 transition-all",
                          item.active
                            ? "bg-gray-200 text-gray-900 dark:bg-gray-800 dark:text-gray-50"
                            : "text-gray-500 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-50"
                        )}
                      >
                        <item.icon className="h-4 w-4" />
                        {item.title}
                      </Link>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </nav>
        </div>
      </div>
    </div>
  )
}
