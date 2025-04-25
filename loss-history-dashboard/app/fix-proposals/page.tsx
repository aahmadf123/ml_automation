"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Clock, CheckCircle, XCircle } from "lucide-react"

// Simple placeholder component that doesn't use MUI
export default function FixProposalsPage() {
  const [proposals] = useState([
    { id: 1, title: "Fix data preprocessing step", status: "approved", date: "2023-11-20" },
    { id: 2, title: "Update feature extraction logic", status: "rejected", date: "2023-11-18" },
    { id: 3, title: "Optimize model training pipeline", status: "pending", date: "2023-11-22" }
  ])

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "approved":
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case "rejected":
        return <XCircle className="h-4 w-4 text-red-500" />
      case "pending":
      default:
        return <Clock className="h-4 w-4 text-amber-500" />
    }
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "approved":
        return <Badge className="bg-green-100 text-green-800">Approved</Badge>
      case "rejected":
        return <Badge className="bg-red-100 text-red-800">Rejected</Badge>
      case "pending":
      default:
        return <Badge className="bg-amber-100 text-amber-800">Pending</Badge>
    }
  }

  return (
    <div className="container mx-auto py-6">
      <h1 className="text-2xl font-bold mb-6">Fix Proposals</h1>
      
      <Card>
        <CardHeader>
          <CardTitle>Recent Fix Proposals</CardTitle>
          <CardDescription>
            Review and manage fix proposals for your ML pipelines
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {proposals.map((proposal) => (
              <div 
                key={proposal.id}
                className="border p-4 rounded-lg flex justify-between items-center"
              >
                <div className="flex items-center gap-2">
                  {getStatusIcon(proposal.status)}
                  <span>{proposal.title}</span>
                </div>
                <div className="flex items-center gap-4">
                  <span className="text-sm text-gray-500">{proposal.date}</span>
                  {getStatusBadge(proposal.status)}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
} 