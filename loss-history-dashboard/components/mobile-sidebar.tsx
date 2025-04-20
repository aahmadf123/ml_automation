"use client"

import { Button } from "@/components/ui/button"

import { useState } from "react"
import { Menu } from "lucide-react"

import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet"
import { DashboardSidebar } from "@/components/dashboard-sidebar"

export function MobileSidebar() {
  const [open, setOpen] = useState(false)

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <Button variant="ghost" size="sm" className="p-0">
          <Menu className="h-5 w-5" />
          <span className="sr-only">Open sidebar</span>
        </Button>
      </SheetTrigger>
      <SheetContent side="left" className="p-0 pt-0 sm:pr-4">
        <SheetHeader className="pl-6 pb-4 pt-4">
          <SheetTitle>Dashboard Menu</SheetTitle>
          <SheetDescription>Navigate through the application</SheetDescription>
        </SheetHeader>
        <DashboardSidebar />
      </SheetContent>
    </Sheet>
  )
}
