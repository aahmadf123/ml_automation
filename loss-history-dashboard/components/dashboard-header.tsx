"use client"

import Link from "next/link"
import { Bell } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { UserButton } from "@/components/ui/user-button"

interface DashboardHeaderProps {
  title?: string
}

export default function DashboardHeader({ title = "Dashboard" }: DashboardHeaderProps) {
  return (
    <header className="sticky top-0 z-50 flex h-16 items-center justify-between border-b bg-background px-4 md:px-6 lg:pl-64">
      <div className="flex items-center gap-2 md:mr-auto">
        <Link href="/" className="flex items-center gap-2 md:hidden">
          <span className="text-xl font-bold">Loss History Agent</span>
          <Badge variant="outline" className="bg-green-500 text-white">
            Live
          </Badge>
        </Link>
        <h1 className="hidden text-xl font-bold md:block">{title}</h1>
      </div>

      <div className="flex items-center gap-4">
        <Button variant="outline" size="sm">
          Test Grouped Notifications
        </Button>

        <div className="relative">
          <Button variant="ghost" size="icon">
            <Bell className="h-5 w-5" />
            <span className="sr-only">Notifications</span>
          </Button>
          <Badge className="absolute -right-1 -top-1 h-5 w-5 rounded-full p-0 text-xs">2</Badge>
        </div>

        <UserButton initials="JD" />
      </div>
    </header>
  )
}
