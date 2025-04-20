"use client"

import type * as React from "react"
import { cn } from "@/lib/utils"

interface NeonCardProps extends React.HTMLAttributes<HTMLDivElement> {
  glowColor?: string
  pulseOnActivity?: boolean
  isActive?: boolean
}

export function NeonCard({
  className,
  children,
  glowColor = "hsl(var(--neon-cyan))",
  pulseOnActivity = false,
  isActive = false,
  ...props
}: NeonCardProps) {
  return (
    <div
      className={cn(
        "relative rounded-md border border-border bg-card text-card-foreground shadow transition-colors",
        "before:absolute before:inset-0 before:rounded-md before:border-2 before:border-transparent before:transition-colors",
        "hover:border-primary data-[state=open]:bg-accent data-[state=open]:text-accent-foreground",
        "dark:focus-within:ring-white/5 focus-within:ring-primary focus-within:ring-2 focus-within:ring-offset-2",
        "before:pointer-events-none",
        "before:transition-colors",
        "before:duration-300",
        "before:ease-in-out",
        "before:z-0",
        "relative z-10",
        className,
        pulseOnActivity && isActive ? "before:animate-pulse-glow" : "",
      )}
      style={{
        "--glow-color": glowColor,
      }}
      {...props}
    >
      {children}
    </div>
  )
}
