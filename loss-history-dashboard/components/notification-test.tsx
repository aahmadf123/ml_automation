"use client"

import { Button } from "@/components/ui/button"
import { toast } from "@/components/ui/toast"

export function NotificationTest() {
  // Update the createTestNotifications function to include priority levels
  const createTestNotifications = () => {
    // Create a group of data quality notifications with different priorities
    toast.error("Missing required field", {
      persistent: true,
      group: "Data Quality Issues",
      source: "Data Validation",
      priority: "high", // High priority
      description: "The 'policy_number' field is missing in 24 records",
    })

    toast.error("Invalid date format", {
      persistent: true,
      group: "Data Quality Issues",
      source: "Data Validation",
      priority: "medium", // Medium priority
      description: "The 'effective_date' field has invalid format in 12 records",
    })

    toast.warning("Potential duplicate records", {
      persistent: true,
      group: "Data Quality Issues",
      source: "Data Validation",
      priority: "low", // Low priority
      description: "Found 8 potentially duplicate policy records",
    })

    // Create a critical notification
    toast.error("Database connection failure", {
      persistent: true,
      group: "System Alerts",
      source: "Database",
      priority: "critical", // Critical priority
      description: "Unable to connect to the primary database. System is operating in fallback mode.",
    })

    // Create a group of system notifications
    toast.info("System maintenance scheduled", {
      persistent: true,
      group: "System Notifications",
      source: "System Admin",
      priority: "medium", // Medium priority
      description: "Scheduled maintenance on Saturday at 2:00 AM EST",
    })

    toast.info("New feature available", {
      persistent: true,
      group: "System Notifications",
      source: "System Admin",
      priority: "low", // Low priority
      description: "Check out the new data visualization tools",
    })

    // Create some ungrouped notifications that will be auto-grouped
    toast.success("Model training complete", {
      persistent: true,
      source: "ML Pipeline",
      priority: "low", // Low priority
    })

    toast.success("Prediction batch processed", {
      persistent: true,
      source: "ML Pipeline",
      priority: "low", // Low priority
    })

    toast.error("API connection failed", {
      persistent: true,
      source: "External Services",
      priority: "high", // High priority
    })
  }

  return (
    <Button onClick={createTestNotifications} variant="outline" size="sm">
      Test Grouped Notifications
    </Button>
  )
}
