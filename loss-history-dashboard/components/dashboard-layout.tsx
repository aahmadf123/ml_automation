import { ReactNode, useState } from "react"
import { DashboardHeader } from "./dashboard-header"
import { DashboardSidebar } from "./dashboard-sidebar"
import { MobileSidebar } from "./mobile-sidebar"
import { Breadcrumbs } from "./breadcrumbs"
import { useWebSocket } from "@/lib/websocket"
import { Badge } from "./ui/badge"
import { Button } from "./ui/button"
import { Bell, RefreshCw, Sun, Moon } from "lucide-react"
import { useTheme } from "next-themes"
import { cn } from "@/lib/utils"

interface DashboardLayoutProps {
  children: ReactNode
  title: string
  description?: string
  actions?: ReactNode
  showWebSocketStatus?: boolean
  showNotifications?: boolean
  showThemeToggle?: boolean
}

export function DashboardLayout({
  children,
  title,
  description,
  actions,
  showWebSocketStatus = true,
  showNotifications = true,
  showThemeToggle = true,
}: DashboardLayoutProps) {
  const { connected, reconnect } = useWebSocket()
  const { theme, setTheme } = useTheme()
  const [notifications, setNotifications] = useState<any[]>([])
  const [showNotificationPanel, setShowNotificationPanel] = useState(false)

  const handleThemeToggle = () => {
    setTheme(theme === "dark" ? "light" : "dark")
  }

  return (
    <div className="flex min-h-screen flex-col">
      <DashboardHeader />
      <MobileSidebar />
      <DashboardSidebar />
      
      <main className="flex-1 p-4 space-y-4 md:p-6 md:space-y-6 lg:ml-64">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">{title}</h1>
            {description && (
              <p className="text-muted-foreground">{description}</p>
            )}
          </div>
          
          <div className="flex items-center gap-2">
            {showWebSocketStatus && (
              <Badge
                variant={connected ? "success" : "destructive"}
                className="cursor-pointer"
                onClick={reconnect}
              >
                {connected ? "Connected" : "Disconnected"}
              </Badge>
            )}

            {showNotifications && (
              <div className="relative">
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setShowNotificationPanel(!showNotificationPanel)}
                >
                  <Bell className="h-5 w-5" />
                  {notifications.length > 0 && (
                    <Badge
                      variant="destructive"
                      className="absolute -top-1 -right-1 h-5 w-5 rounded-full p-0 flex items-center justify-center"
                    >
                      {notifications.length}
                    </Badge>
                  )}
                </Button>

                {showNotificationPanel && (
                  <div className="absolute right-0 mt-2 w-80 bg-background border rounded-lg shadow-lg z-50">
                    <div className="p-4">
                      <h3 className="font-semibold mb-2">Notifications</h3>
                      {notifications.length === 0 ? (
                        <p className="text-muted-foreground text-sm">No new notifications</p>
                      ) : (
                        <div className="space-y-2">
                          {notifications.map((notification, index) => (
                            <div
                              key={index}
                              className="p-2 rounded-md bg-muted hover:bg-muted/80 cursor-pointer"
                            >
                              <p className="text-sm font-medium">{notification.title}</p>
                              <p className="text-xs text-muted-foreground">
                                {notification.message}
                              </p>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}

            {showThemeToggle && (
              <Button
                variant="ghost"
                size="icon"
                onClick={handleThemeToggle}
                className="ml-2"
              >
                {theme === "dark" ? (
                  <Sun className="h-5 w-5" />
                ) : (
                  <Moon className="h-5 w-5" />
                )}
              </Button>
            )}

            {actions && (
              <div className="flex items-center gap-2 ml-4">
                {actions}
              </div>
            )}
          </div>
        </div>

        <Breadcrumbs />

        <div className={cn(
          "space-y-4",
          !connected && "opacity-50 pointer-events-none"
        )}>
          {children}
        </div>
      </main>
    </div>
  )
} 