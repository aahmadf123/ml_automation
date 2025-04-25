import type { Metadata } from "next"
import { Inter } from "next/font/google"
import "../styles/globals.css"
import { DashboardSidebar } from "@/components/dashboard-sidebar"
import { ThemeProvider } from "@/components/theme-provider"
import { Toaster } from "@/components/ui/toast"

const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "Loss History Dashboard",
  description: "ML Pipeline and Model Performance Dashboard",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <ThemeProvider>
          <div className="min-h-screen overflow-hidden flex">
            <DashboardSidebar />
            <main className="flex-1 w-full max-w-screen-2xl mx-auto">{children}</main>
          </div>
          <Toaster />
        </ThemeProvider>
      </body>
    </html>
  )
}
