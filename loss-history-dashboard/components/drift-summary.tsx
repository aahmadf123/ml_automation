"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { MiniSparkline } from "@/components/mini-sparkline"
import { useState } from "react"
import { AlertTriangle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useToast } from "@/hooks/use-toast"

// Mock data for feature drift
const featureDriftData = [
  {
    name: "property_value",
    current: 5.5,
    threshold: 5,
    history: [1.2, 1.8, 2.1, 3.5, 5.5],
    trend: "up",
    warning: true,
  },
  {
    name: "claim_amount",
    current: 3.7,
    threshold: 5,
    history: [4.2, 3.9, 3.5, 3.6, 3.7],
    trend: "down",
  },
  {
    name: "property_age",
    current: 1.3,
    threshold: 3,
    history: [1.5, 1.4, 1.3, 1.3, 1.3],
    trend: "stable",
  },
  {
    name: "location_risk",
    current: 6.2,
    threshold: 5,
    history: [4.8, 5.1, 5.5, 5.9, 6.2],
    trend: "up",
    warning: true,
  },
]

export function DriftSummary() {
  const { toast } = useToast()
  const [selectedFeature, setSelectedFeature] = useState<string | null>(null)

  const handleManageThresholds = () => {
    // In a real app, this would navigate to the threshold management page
    // For now, we'll just show a toast
    toast({
      title: "Manage Thresholds",
      description: "Navigating to threshold management page...",
    })
    // In a real app: router.push("/threshold-management")
    window.location.href = "/threshold-management"
  }

  return (
    <Card className="shadow-md">
      <CardHeader className="pb-2">
        <div className="flex justify-between items-center">
          <div>
            <CardTitle>Feature Drift Summary</CardTitle>
            <CardDescription>Current drift percentages compared to thresholds</CardDescription>
          </div>
          <Button variant="outline" size="sm" onClick={handleManageThresholds}>
            Manage Thresholds
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {featureDriftData.map((feature) => (
            <Card
              key={feature.name}
              className={`overflow-hidden transition-all duration-200 hover:shadow-lg cursor-pointer ${
                feature.warning
                  ? "border-red-400 dark:border-red-500"
                  : "hover:border-teal-400 dark:hover:border-teal-500"
              }`}
              onClick={() => setSelectedFeature(feature.name === selectedFeature ? null : feature.name)}
            >
              <CardContent className="p-4">
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <h3 className="font-medium text-sm capitalize">{feature.name.replace(/_/g, " ")}</h3>
                    <div className="flex items-center mt-1">
                      <span
                        className={`text-xl font-bold ${
                          feature.current > feature.threshold
                            ? "text-red-500 dark:text-red-400"
                            : "text-green-500 dark:text-green-400"
                        }`}
                      >
                        {feature.current}%
                      </span>
                      <span className="text-muted-foreground text-sm ml-2">/ {feature.threshold}%</span>
                    </div>
                  </div>
                  <div
                    className={`h-2 w-2 rounded-full mt-1 ${
                      feature.trend === "up"
                        ? "bg-red-500"
                        : feature.trend === "down"
                          ? "bg-green-500"
                          : "bg-yellow-500"
                    }`}
                  />
                </div>

                <div className="h-10 mt-2">
                  <MiniSparkline
                    data={feature.history}
                    color={feature.current > feature.threshold ? "#f87171" : "#10b981"}
                  />
                </div>

                {feature.warning && (
                  <div className="mt-2 flex items-center text-xs text-red-500">
                    <AlertTriangle className="h-3 w-3 mr-1" />
                    <span>Exceeds threshold</span>
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
        </div>

        {selectedFeature && (
          <div className="mt-6 p-4 border rounded-lg">
            <h3 className="font-medium mb-2 capitalize">{selectedFeature.replace(/_/g, " ")} Details</h3>
            <p className="text-sm text-muted-foreground mb-4">
              {featureDriftData.find((f) => f.name === selectedFeature)?.warning
                ? "This feature is exceeding its drift threshold and may require attention."
                : "This feature is within acceptable drift parameters."}
            </p>
            <div className="flex space-x-2">
              <Button
                className="px-3 py-1.5 text-sm bg-primary text-primary-foreground rounded-md hover:bg-primary/90"
                onClick={() => {
                  toast({
                    title: "Self-Heal Initiated",
                    description: `Self-healing process started for ${selectedFeature.replace(/_/g, " ")}`,
                  })
                }}
              >
                Self-Heal
              </Button>
              <Button
                className="px-3 py-1.5 text-sm bg-secondary text-secondary-foreground rounded-md hover:bg-secondary/90"
                onClick={() => {
                  toast({
                    title: "Fix Proposed",
                    description: `Fix proposal created for ${selectedFeature.replace(/_/g, " ")}`,
                  })
                }}
              >
                Propose Fix
              </Button>
              <Button
                variant="outline"
                className="px-3 py-1.5 text-sm rounded-md"
                onClick={() => handleManageThresholds()}
              >
                Adjust Threshold
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
