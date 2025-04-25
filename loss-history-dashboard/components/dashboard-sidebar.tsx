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
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { useState } from "react"
import { cn } from "@/lib/utils"

interface NavItem {
  title: string
  href: string
  icon: any
  active: boolean
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
        },
      ],
    },
    {
      title: "Analytics",
      items: [
        {
          title: "Business Insights",
          href: "/business-insights",
          icon: TrendingUp,
          active: isActive("/business-insights"),
        },
        {
          title: "Visualizations",
          href: "/visualizations",
          icon: BarChart3,
          active: isActive("/visualizations"),
        },
      ],
    },
    {
      title: "Monitoring",
      items: [
        {
          title: "Pipeline Health",
          href: "/pipeline-health",
          icon: Gauge,
          active: isActive("/pipeline-health"),
        },
        {
          title: "Drift Monitor",
          href: "/drift-monitor",
          icon: LineChart,
          active: isActive("/drift-monitor"),
        },
        {
          title: "Data Quality",
          href: "/data-quality",
          icon: FileWarning,
          active: isActive("/data-quality"),
        },
        {
          title: "Incidents",
          href: "/incidents",
          icon: Bell,
          active: isActive("/incidents"),
        },
      ],
    },
    {
      title: "Models",
      items: [
        {
          title: "Model Metrics",
          href: "/model-metrics",
          icon: Gauge,
          active: isActive("/model-metrics"),
        },
        {
          title: "Model Comparison",
          href: "/model-comparison",
          icon: GitCompare,
          active: isActive("/model-comparison"),
        },
        {
          title: "Model Explainability",
          href: "/model-explainability",
          icon: Lightbulb,
          active: isActive("/model-explainability"),
        },
      ],
    },
    {
      title: "Data",
      items: [
        {
          title: "Data Ingestion",
          href: "/data-ingestion",
          icon: Database,
          active: isActive("/data-ingestion"),
        },
      ],
    },
    {
      title: "Management",
      items: [
        {
          title: "Fix Templates",
          href: "/fix-templates",
          icon: Workflow,
          active: isActive("/fix-templates"),
        },
        {
          title: "Code Interpreter",
          href: "/code-interpreter",
          icon: Code,
          active: isActive("/code-interpreter"),
        },
        {
          title: "Settings",
          href: "/settings",
          icon: Settings,
          active: isActive("/settings"),
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
