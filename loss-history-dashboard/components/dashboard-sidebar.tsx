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
} from "lucide-react"
import { Badge } from "@/components/ui/badge"

export function DashboardSidebar() {
  const pathname = usePathname()

  const isActive = (path: string) => {
    return pathname === path
  }

  const navItems = [
    {
      title: "Dashboard",
      href: "/",
      icon: Home,
      active: isActive("/"),
    },
    {
      title: "Claims",
      href: "/claims",
      icon: Database,
      active: isActive("/claims"),
    },
    {
      title: "Model Metrics",
      href: "/model-metrics",
      icon: Gauge,
      active: isActive("/model-metrics"),
    },
    {
      title: "Drift Monitor",
      href: "/drift-monitor",
      icon: LineChart,
      active: isActive("/drift-monitor"),
    },
    {
      title: "Threshold Management",
      href: "/threshold-management",
      icon: AlertTriangle,
      active: isActive("/threshold-management"),
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
    {
      title: "Data Quality",
      href: "/data-quality",
      icon: FileWarning,
      active: isActive("/data-quality"),
    },
    {
      title: "Data Ingestion",
      href: "/data-ingestion",
      icon: Database,
      active: isActive("/data-ingestion"),
    },
    {
      title: "Visualizations",
      href: "/visualizations",
      icon: BarChart3,
      active: isActive("/visualizations"),
    },
    {
      title: "Incidents",
      href: "/incidents",
      icon: Bell,
      active: isActive("/incidents"),
    },
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
      title: "Notification Analytics",
      href: "/notification-analytics",
      icon: Bell,
      active: isActive("/notification-analytics"),
    },
    {
      title: "Settings",
      href: "/settings",
      icon: Settings,
      active: isActive("/settings"),
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
            {navItems.map((item, index) => (
              <Link
                key={index}
                href={item.href}
                className={`flex items-center gap-3 rounded-lg px-3 py-2 transition-all ${
                  item.active
                    ? "bg-gray-200 text-gray-900 dark:bg-gray-800 dark:text-gray-50"
                    : "text-gray-500 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-50"
                }`}
              >
                <item.icon className="h-4 w-4" />
                {item.title}
              </Link>
            ))}
          </nav>
        </div>
      </div>
    </div>
  )
}
