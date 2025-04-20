"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { formatDistanceToNow } from "date-fns"
import { ChevronRight, Search } from "lucide-react"
import { Input } from "@/components/ui/input"

interface Claim {
  id: string
  customer: string
  amount: number
  date: string
  status: "approved" | "pending" | "denied"
  type: string
}

export function RecentClaims() {
  const [searchTerm, setSearchTerm] = useState("")

  const claims: Claim[] = [
    {
      id: "CLM-1001",
      customer: "John Smith",
      amount: 12500,
      date: new Date(Date.now() - 2 * 86400000).toISOString(),
      status: "approved",
      type: "Water Damage",
    },
    {
      id: "CLM-1002",
      customer: "Sarah Johnson",
      amount: 8750,
      date: new Date(Date.now() - 3 * 86400000).toISOString(),
      status: "pending",
      type: "Fire",
    },
    {
      id: "CLM-1003",
      customer: "Michael Brown",
      amount: 5200,
      date: new Date(Date.now() - 5 * 86400000).toISOString(),
      status: "denied",
      type: "Theft",
    },
    {
      id: "CLM-1004",
      customer: "Emily Davis",
      amount: 15800,
      date: new Date(Date.now() - 7 * 86400000).toISOString(),
      status: "approved",
      type: "Wind",
    },
    {
      id: "CLM-1005",
      customer: "Robert Wilson",
      amount: 9300,
      date: new Date(Date.now() - 8 * 86400000).toISOString(),
      status: "pending",
      type: "Water Damage",
    },
    {
      id: "CLM-1006",
      customer: "Jennifer Lee",
      amount: 7600,
      date: new Date(Date.now() - 10 * 86400000).toISOString(),
      status: "approved",
      type: "Liability",
    },
  ]

  const filteredClaims = claims.filter(
    (claim) =>
      claim.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      claim.customer.toLowerCase().includes(searchTerm.toLowerCase()) ||
      claim.type.toLowerCase().includes(searchTerm.toLowerCase()),
  )

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-lg flex items-center justify-between">Recent Claims</CardTitle>
        <div className="relative mt-2">
          <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search claims..."
            className="pl-8"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <div className="h-[300px] overflow-y-auto scrollbar-thin scrollbar-thumb-rounded scrollbar-thumb-gray-300">
          <div className="divide-y">
            {filteredClaims.map((claim) => (
              <div key={claim.id} className="flex items-center justify-between p-4 hover:bg-muted/50 transition-colors">
                <div className="flex items-center space-x-4">
                  <div className="space-y-1">
                    <div className="flex items-center">
                      <p className="font-medium">{claim.customer}</p>
                      <span className="text-xs text-muted-foreground ml-2">({claim.id})</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Badge variant="outline">{claim.type}</Badge>
                      <p className="text-sm text-muted-foreground">
                        {formatDistanceToNow(new Date(claim.date), { addSuffix: true })}
                      </p>
                    </div>
                  </div>
                </div>
                <div className="flex items-center space-x-4">
                  <div className="text-right">
                    <p className="font-medium">${claim.amount.toLocaleString()}</p>
                    <Badge
                      variant={
                        claim.status === "approved" ? "success" : claim.status === "denied" ? "destructive" : "outline"
                      }
                    >
                      {claim.status}
                    </Badge>
                  </div>
                  <ChevronRight className="h-4 w-4 text-muted-foreground" />
                </div>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
