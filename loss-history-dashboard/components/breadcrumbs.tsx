"use client"

import { ChevronRight, Home } from "lucide-react"
import { usePathname } from "next/navigation"
import Link from "next/link"

export function Breadcrumbs() {
  const pathname = usePathname()

  // Skip rendering breadcrumbs on the home page
  if (pathname === "/") {
    return null
  }

  // Split the pathname into segments and remove empty strings
  const segments = pathname.split("/").filter(Boolean)

  return (
    <nav className="flex items-center text-sm text-muted-foreground">
      <Link href="/" className="flex items-center hover:text-foreground">
        <Home className="mr-1 h-4 w-4" />
        Home
      </Link>

      {segments.map((segment, index) => {
        // Create the path up to this segment
        const path = `/${segments.slice(0, index + 1).join("/")}`

        // Format the segment for display (capitalize, replace hyphens with spaces)
        const formattedSegment = segment.replace(/-/g, " ").replace(/\b\w/g, (char) => char.toUpperCase())

        return (
          <div key={path} className="flex items-center">
            <ChevronRight className="mx-1 h-4 w-4" />
            <Link
              href={path}
              className={index === segments.length - 1 ? "font-medium text-foreground" : "hover:text-foreground"}
            >
              {formattedSegment}
            </Link>
          </div>
        )
      })}
    </nav>
  )
}
