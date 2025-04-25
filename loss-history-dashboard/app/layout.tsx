import type React from "react"
import "@/app/globals.css"
import { DashboardSidebar } from "@/components/dashboard-sidebar"
import { ThemeProvider } from "@/components/theme-provider"
import { Toaster } from "@/components/ui/toast"
import { Inter } from "next/font/google"

const inter = Inter({ subsets: ["latin"] })

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <ThemeProvider>
          <div className="min-h-screen overflow-hidden">
            <DashboardSidebar />
            <main className="flex-1 w-full lg:ml-64">{children}</main>
          </div>
          <Toaster />
        </ThemeProvider>
      </body>
    </html>
  )
}

export const metadata = {
      title: 'Model Performance & Health Dashboard',
      description: 'Monitor your production models with comprehensive analytics and interactive visualizations'
    };
