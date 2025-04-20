"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { NotificationData } from "@/lib/notification-data"
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
  Tooltip,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

interface NotificationDistributionChartProps {
  data: NotificationData[]
}

export function NotificationDistributionChart({ data }: NotificationDistributionChartProps) {
  // Prepare data for priority distribution
  const priorityData = [
    { name: "Critical", value: data.filter((n) => n.priority === "critical").length, color: "#9333ea" },
    { name: "High", value: data.filter((n) => n.priority === "high").length, color: "#ef4444" },
    { name: "Medium", value: data.filter((n) => n.priority === "medium").length, color: "#f59e0b" },
    { name: "Low", value: data.filter((n) => n.priority === "low").length, color: "#3b82f6" },
  ]

  // Prepare data for type distribution
  const typeData = [
    { name: "Error", value: data.filter((n) => n.type === "error").length, color: "#ef4444" },
    { name: "Warning", value: data.filter((n) => n.type === "warning").length, color: "#f59e0b" },
    { name: "Info", value: data.filter((n) => n.type === "info").length, color: "#3b82f6" },
    { name: "Success", value: data.filter((n) => n.type === "success").length, color: "#10b981" },
  ]

  // Prepare data for source distribution
  const sourceData = Object.entries(
    data.reduce(
      (acc, notification) => {
        acc[notification.source] = (acc[notification.source] || 0) + 1
        return acc
      },
      {} as Record<string, number>,
    ),
  )
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 10) // Top 10 sources

  // Custom tooltip for pie chart
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-background border rounded-md shadow-md p-2 text-sm">
          <p className="font-medium">{`${payload[0].name}: ${payload[0].value}`}</p>
          <p className="text-xs text-muted-foreground">{`${Math.round((payload[0].value / data.length) * 100)}%`}</p>
        </div>
      )
    }
    return null
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Notification Distribution</CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="priority">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="priority">By Priority</TabsTrigger>
            <TabsTrigger value="type">By Type</TabsTrigger>
            <TabsTrigger value="source">By Source</TabsTrigger>
          </TabsList>

          <TabsContent value="priority" className="h-[350px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={priorityData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={2}
                  dataKey="value"
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  labelLine={false}
                >
                  {priorityData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </TabsContent>

          <TabsContent value="type" className="h-[350px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={typeData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={2}
                  dataKey="value"
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  labelLine={false}
                >
                  {typeData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </TabsContent>

          <TabsContent value="source" className="h-[350px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={sourceData} layout="vertical" margin={{ top: 5, right: 30, left: 100, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                <XAxis type="number" />
                <YAxis type="category" dataKey="name" width={100} />
                <Tooltip />
                <Bar dataKey="value" fill="#4ECDC4" />
              </BarChart>
            </ResponsiveContainer>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
