"use client"

import type React from "react"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { formatDistanceToNow } from "date-fns"
import { User, FileText, AlertTriangle, CheckCircle, Settings, Upload, Download, Eye } from "lucide-react"

interface ActivityItem {
  id: number
  action: string
  user: string
  timestamp: string
  icon: React.ReactNode
  iconBg: string
}

export function ActivityFeed() {
  const activities: ActivityItem[] = [
    {
      id: 1,
      action: "Updated model parameters",
      user: "System",
      timestamp: new Date(Date.now() - 15 * 60000).toISOString(),
      icon: <Settings className="h-4 w-4 text-white" />,
      iconBg: "bg-blue-500",
    },
    {
      id: 2,
      action: "Uploaded new training data",
      user: "Maria Garcia",
      timestamp: new Date(Date.now() - 45 * 60000).toISOString(),
      icon: <Upload className="h-4 w-4 text-white" />,
      iconBg: "bg-green-500",
    },
    {
      id: 3,
      action: "Reviewed anomaly report",
      user: "David Kim",
      timestamp: new Date(Date.now() - 2 * 3600000).toISOString(),
      icon: <Eye className="h-4 w-4 text-white" />,
      iconBg: "bg-purple-500",
    },
    {
      id: 4,
      action: "Exported monthly metrics",
      user: "Alex Johnson",
      timestamp: new Date(Date.now() - 5 * 3600000).toISOString(),
      icon: <Download className="h-4 w-4 text-white" />,
      iconBg: "bg-amber-500",
    },
    {
      id: 5,
      action: "Approved claim override",
      user: "Samantha Lee",
      timestamp: new Date(Date.now() - 8 * 3600000).toISOString(),
      icon: <CheckCircle className="h-4 w-4 text-white" />,
      iconBg: "bg-emerald-500",
    },
    {
      id: 6,
      action: "Flagged suspicious pattern",
      user: "System",
      timestamp: new Date(Date.now() - 12 * 3600000).toISOString(),
      icon: <AlertTriangle className="h-4 w-4 text-white" />,
      iconBg: "bg-red-500",
    },
    {
      id: 7,
      action: "Updated documentation",
      user: "James Wilson",
      timestamp: new Date(Date.now() - 24 * 3600000).toISOString(),
      icon: <FileText className="h-4 w-4 text-white" />,
      iconBg: "bg-indigo-500",
    },
  ]

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">Activity Feed</CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <div className="h-[300px] overflow-y-auto px-6 py-2 scrollbar-thin scrollbar-thumb-rounded scrollbar-thumb-gray-300">
          <div className="relative">
            {/* Timeline line */}
            <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-muted" />

            <div className="space-y-6 pt-2">
              {activities.map((activity) => (
                <div key={activity.id} className="relative flex items-start ml-6">
                  {/* Timeline dot */}
                  <div
                    className={`absolute -left-10 flex h-8 w-8 items-center justify-center rounded-full ${activity.iconBg}`}
                  >
                    {activity.icon}
                  </div>

                  <div className="flex flex-col">
                    <p className="font-medium">{activity.action}</p>
                    <div className="flex items-center text-sm text-muted-foreground">
                      <User className="mr-1 h-3 w-3" />
                      <span>{activity.user}</span>
                      <span className="mx-1">•</span>
                      <span>{formatDistanceToNow(new Date(activity.timestamp), { addSuffix: true })}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
