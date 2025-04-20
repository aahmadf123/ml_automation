"use client"

import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { CheckCircle, XCircle, Clock, FileText, MoreHorizontal } from "lucide-react"
import { formatDistanceToNow } from "date-fns"

export function IngestionHistory() {
  // Mock data for ingestion history
  const historyData = [
    {
      id: "ING-1234",
      fileName: "loss_history_2025_q1.csv",
      status: "completed",
      recordCount: 12500,
      startTime: new Date(Date.now() - 3600000 * 2), // 2 hours ago
      endTime: new Date(Date.now() - 3600000 * 1.5), // 1.5 hours ago
      user: "Jane Doe",
    },
    {
      id: "ING-1233",
      fileName: "property_data_update.xlsx",
      status: "failed",
      recordCount: 5200,
      startTime: new Date(Date.now() - 3600000 * 24), // 1 day ago
      endTime: new Date(Date.now() - 3600000 * 23.8), // 23.8 hours ago
      user: "John Smith",
    },
    {
      id: "ING-1232",
      fileName: "claim_history_2024_q4.csv",
      status: "completed",
      recordCount: 8750,
      startTime: new Date(Date.now() - 3600000 * 48), // 2 days ago
      endTime: new Date(Date.now() - 3600000 * 47.5), // 47.5 hours ago
      user: "Jane Doe",
    },
    {
      id: "ING-1231",
      fileName: "location_risk_factors.json",
      status: "completed",
      recordCount: 3200,
      startTime: new Date(Date.now() - 3600000 * 72), // 3 days ago
      endTime: new Date(Date.now() - 3600000 * 71.2), // 71.2 hours ago
      user: "Alex Johnson",
    },
    {
      id: "ING-1230",
      fileName: "policy_updates_march.csv",
      status: "processing",
      recordCount: 4500,
      startTime: new Date(Date.now() - 3600000 * 1), // 1 hour ago
      endTime: null,
      user: "John Smith",
    },
  ]

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "completed":
        return (
          <Badge className="bg-green-500">
            <CheckCircle className="mr-1 h-3 w-3" />
            Completed
          </Badge>
        )
      case "failed":
        return (
          <Badge variant="destructive">
            <XCircle className="mr-1 h-3 w-3" />
            Failed
          </Badge>
        )
      case "processing":
        return (
          <Badge variant="outline" className="bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-400">
            <Clock className="mr-1 h-3 w-3 animate-spin" />
            Processing
          </Badge>
        )
      default:
        return <Badge variant="outline">{status}</Badge>
    }
  }

  const formatDuration = (startTime: Date, endTime: Date | null) => {
    if (!endTime) return "In progress"

    const durationMs = endTime.getTime() - startTime.getTime()
    const minutes = Math.floor(durationMs / 60000)
    const seconds = Math.floor((durationMs % 60000) / 1000)

    return `${minutes}m ${seconds}s`
  }

  return (
    <div className="rounded-md border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>ID</TableHead>
            <TableHead>File Name</TableHead>
            <TableHead>Status</TableHead>
            <TableHead>Records</TableHead>
            <TableHead>Duration</TableHead>
            <TableHead>Started</TableHead>
            <TableHead>User</TableHead>
            <TableHead className="text-right">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {historyData.map((item) => (
            <TableRow key={item.id}>
              <TableCell className="font-medium">{item.id}</TableCell>
              <TableCell>{item.fileName}</TableCell>
              <TableCell>{getStatusBadge(item.status)}</TableCell>
              <TableCell>{item.recordCount.toLocaleString()}</TableCell>
              <TableCell>{formatDuration(item.startTime, item.endTime)}</TableCell>
              <TableCell>{formatDistanceToNow(item.startTime, { addSuffix: true })}</TableCell>
              <TableCell>{item.user}</TableCell>
              <TableCell className="text-right">
                <Button variant="ghost" size="icon">
                  <FileText className="h-4 w-4" />
                  <span className="sr-only">View details</span>
                </Button>
                <Button variant="ghost" size="icon">
                  <MoreHorizontal className="h-4 w-4" />
                  <span className="sr-only">More options</span>
                </Button>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}
