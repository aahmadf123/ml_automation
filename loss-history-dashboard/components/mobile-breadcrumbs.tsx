"use client"

import { useBreadcrumbs } from "@/hooks/use-breadcrumbs"
import { cn } from "@/lib/utils"
import { ChevronRight } from "lucide-react"
import Link from "next/link"

interface MobileBreadcrumbsProps {
  className?: string
}

export function MobileBreadcrumbs({ className }: MobileBreadcrumbsProps) {
  const breadcrumbs = useBreadcrumbs()

  if (breadcrumbs.length <= 1) {
    return null
  }

  // For mobile, we only show the last two breadcrumbs to save space
  const displayBreadcrumbs = breadcrumbs.length > 2 ? [breadcrumbs[0], ...breadcrumbs.slice(-1)] : breadcrumbs

  return (
    <nav
      aria-label="Breadcrumb"
      className={cn("md:hidden flex items-center text-sm overflow-x-auto py-2 scrollbar-hide", className)}
    >
      <ol className="flex items-center space-x-1">
        {displayBreadcrumbs.map((breadcrumb, index) => (
          <li key={breadcrumb.href} className="flex items-center whitespace-nowrap">
            {index > 0 && <ChevronRight className="mx-1 h-3 w-3 text-muted-foreground" />}
            <Link
              href={breadcrumb.href}
              className={cn(
                "text-muted-foreground hover:text-foreground transition-colors",
                breadcrumb.active && "text-foreground font-medium",
              )}
              aria-current={breadcrumb.active ? "page" : undefined}
            >
              {breadcrumb.label}
            </Link>
          </li>
        ))}
      </ol>
    </nav>
  )
}
