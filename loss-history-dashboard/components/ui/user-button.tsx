"use client"

import { Button } from "@/components/ui/button"

interface UserButtonProps {
  initials: string
}

export function UserButton({ initials }: UserButtonProps) {
  return (
    <Button variant="ghost" className="relative h-8 w-8 rounded-full bg-primary text-primary-foreground">
      {initials}
    </Button>
  )
}
