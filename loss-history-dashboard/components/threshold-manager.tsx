"use client"

import { useState } from "react"
import { Plus, Save, Trash2, AlertTriangle, Bell, BellOff } from "lucide-react"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { useToast } from "@/hooks/use-toast"

// Types for threshold configuration
export interface ThresholdConfig {
  id: string
  name: string
  metricType: "drift" | "performance"
  metricName: string
  operator: ">" | "<" | ">=" | "<=" | "="
  value: number
  severity: "info" | "warning" | "critical"
  enabled: boolean
  notificationChannels: string[]
  createdAt: string
  updatedAt: string
}

// Mock data for available metrics
const availableMetrics = {
  drift: [
    { id: "drift_property_value", name: "Property Value Drift" },
    { id: "drift_claim_amount", name: "Claim Amount Drift" },
    { id: "drift_property_age", name: "Property Age Drift" },
    { id: "drift_location_risk", name: "Location Risk Drift" },
    { id: "drift_overall", name: "Overall Distribution Drift" },
  ],
  performance: [
    { id: "perf_rmse", name: "RMSE" },
    { id: "perf_mae", name: "MAE" },
    { id: "perf_mse", name: "MSE" },
    { id: "perf_r2", name: "R² Score" },
    { id: "perf_accuracy", name: "Accuracy" },
    { id: "perf_f1", name: "F1 Score" },
    { id: "perf_precision", name: "Precision" },
    { id: "perf_recall", name: "Recall" },
    { id: "perf_auc", name: "AUC-ROC" },
  ],
}

// Mock data for notification channels
const notificationChannels = [
  { id: "email", name: "Email" },
  { id: "slack", name: "Slack" },
  { id: "dashboard", name: "Dashboard" },
  { id: "webhook", name: "Webhook" },
]

// Mock initial thresholds
const initialThresholds: ThresholdConfig[] = [
  {
    id: "1",
    name: "High Property Value Drift",
    metricType: "drift",
    metricName: "drift_property_value",
    operator: ">",
    value: 5,
    severity: "warning",
    enabled: true,
    notificationChannels: ["email", "dashboard"],
    createdAt: "2023-04-15T09:30:00",
    updatedAt: "2023-04-15T09:30:00",
  },
  {
    id: "2",
    name: "Critical RMSE Alert",
    metricType: "performance",
    metricName: "perf_rmse",
    operator: ">",
    value: 0.5,
    severity: "critical",
    enabled: true,
    notificationChannels: ["email", "slack", "dashboard"],
    createdAt: "2023-04-10T14:20:00",
    updatedAt: "2023-04-12T11:15:00",
  },
  {
    id: "3",
    name: "Low R² Score",
    metricType: "performance",
    metricName: "perf_r2",
    operator: "<",
    value: 0.8,
    severity: "info",
    enabled: false,
    notificationChannels: ["dashboard"],
    createdAt: "2023-04-05T16:45:00",
    updatedAt: "2023-04-05T16:45:00",
  },
]

export function ThresholdManager() {
  const { toast } = useToast()
  const [thresholds, setThresholds] = useState<ThresholdConfig[]>(initialThresholds)
  const [activeTab, setActiveTab] = useState<"drift" | "performance">("drift")
  const [isCreating, setIsCreating] = useState(false)
  const [editingThreshold, setEditingThreshold] = useState<ThresholdConfig | null>(null)

  // New threshold template
  const newThresholdTemplate: Omit<ThresholdConfig, "id" | "createdAt" | "updatedAt"> = {
    name: "",
    metricType: activeTab,
    metricName: "",
    operator: ">",
    value: 0,
    severity: "warning",
    enabled: true,
    notificationChannels: ["dashboard"],
  }

  const [newThreshold, setNewThreshold] = useState(newThresholdTemplate)

  const handleCreateThreshold = () => {
    if (!newThreshold.name || !newThreshold.metricName) {
      toast({
        title: "Validation Error",
        description: "Please fill in all required fields",
        variant: "destructive",
      })
      return
    }

    const now = new Date().toISOString()
    const threshold: ThresholdConfig = {
      ...newThreshold,
      id: `threshold_${Date.now()}`,
      createdAt: now,
      updatedAt: now,
    }

    setThresholds([...thresholds, threshold])
    setNewThreshold(newThresholdTemplate)
    setIsCreating(false)

    toast({
      title: "Threshold Created",
      description: `Threshold "${threshold.name}" has been created successfully.`,
    })
  }

  const handleUpdateThreshold = () => {
    if (!editingThreshold) return

    const updatedThresholds = thresholds.map((t) =>
      t.id === editingThreshold.id ? { ...editingThreshold, updatedAt: new Date().toISOString() } : t,
    )

    setThresholds(updatedThresholds)
    setEditingThreshold(null)

    toast({
      title: "Threshold Updated",
      description: `Threshold "${editingThreshold.name}" has been updated successfully.`,
    })
  }

  const handleDeleteThreshold = (id: string) => {
    setThresholds(thresholds.filter((t) => t.id !== id))
    if (editingThreshold?.id === id) {
      setEditingThreshold(null)
    }

    toast({
      title: "Threshold Deleted",
      description: "The threshold has been deleted successfully.",
    })
  }

  const handleToggleThreshold = (id: string, enabled: boolean) => {
    const updatedThresholds = thresholds.map((t) =>
      t.id === id ? { ...t, enabled, updatedAt: new Date().toISOString() } : t,
    )
    setThresholds(updatedThresholds)

    const threshold = thresholds.find((t) => t.id === id)
    toast({
      title: enabled ? "Threshold Enabled" : "Threshold Disabled",
      description: `"${threshold?.name}" has been ${enabled ? "enabled" : "disabled"}.`,
    })
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "info":
        return "bg-blue-500 text-white"
      case "warning":
        return "bg-yellow-500 text-white"
      case "critical":
        return "bg-red-500 text-white"
      default:
        return "bg-gray-500 text-white"
    }
  }

  const getOperatorLabel = (operator: string) => {
    switch (operator) {
      case ">":
        return "greater than"
      case "<":
        return "less than"
      case ">=":
        return "greater than or equal to"
      case "<=":
        return "less than or equal to"
      case "=":
        return "equal to"
      default:
        return operator
    }
  }

  const getMetricName = (metricId: string) => {
    const allMetrics = [...availableMetrics.drift, ...availableMetrics.performance]
    return allMetrics.find((m) => m.id === metricId)?.name || metricId
  }

  return (
    <Card className="shadow-md">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Custom Alert Thresholds</span>
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              setIsCreating(true)
              setEditingThreshold(null)
              setNewThreshold({ ...newThresholdTemplate, metricType: activeTab })
            }}
          >
            <Plus className="mr-2 h-4 w-4" />
            New Threshold
          </Button>
        </CardTitle>
        <CardDescription>Configure custom thresholds for model drift and performance metrics</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="drift" onValueChange={(value) => setActiveTab(value as "drift" | "performance")}>
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="drift">Drift Thresholds</TabsTrigger>
            <TabsTrigger value="performance">Performance Thresholds</TabsTrigger>
          </TabsList>

          {["drift", "performance"].map((tabValue) => (
            <TabsContent key={tabValue} value={tabValue} className="space-y-4">
              {isCreating && activeTab === tabValue ? (
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg">Create New Threshold</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-1 gap-4">
                      <div className="space-y-2">
                        <Label htmlFor="threshold-name">Threshold Name</Label>
                        <Input
                          id="threshold-name"
                          placeholder="E.g., High Property Value Drift"
                          value={newThreshold.name}
                          onChange={(e) => setNewThreshold({ ...newThreshold, name: e.target.value })}
                        />
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="metric-name">Metric</Label>
                        <Select
                          value={newThreshold.metricName}
                          onValueChange={(value) => setNewThreshold({ ...newThreshold, metricName: value })}
                        >
                          <SelectTrigger id="metric-name">
                            <SelectValue placeholder="Select a metric" />
                          </SelectTrigger>
                          <SelectContent>
                            {availableMetrics[tabValue as "drift" | "performance"].map((metric) => (
                              <SelectItem key={metric.id} value={metric.id}>
                                {metric.name}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <Label htmlFor="operator">Condition</Label>
                          <Select
                            value={newThreshold.operator}
                            onValueChange={(value) =>
                              setNewThreshold({ ...newThreshold, operator: value as ThresholdConfig["operator"] })
                            }
                          >
                            <SelectTrigger id="operator">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value=">">Greater than (&gt;)</SelectItem>
                              <SelectItem value="<">Less than (&lt;)</SelectItem>
                              <SelectItem value=">=">Greater than or equal to (≥)</SelectItem>
                              <SelectItem value="<=">Less than or equal to (≤)</SelectItem>
                              <SelectItem value="=">Equal to (=)</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>

                        <div className="space-y-2">
                          <Label htmlFor="threshold-value">Value</Label>
                          <Input
                            id="threshold-value"
                            type="number"
                            step="0.01"
                            value={newThreshold.value}
                            onChange={(e) =>
                              setNewThreshold({ ...newThreshold, value: Number.parseFloat(e.target.value) || 0 })
                            }
                          />
                        </div>
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="severity">Severity</Label>
                        <Select
                          value={newThreshold.severity}
                          onValueChange={(value) =>
                            setNewThreshold({ ...newThreshold, severity: value as ThresholdConfig["severity"] })
                          }
                        >
                          <SelectTrigger id="severity">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="info">Info</SelectItem>
                            <SelectItem value="warning">Warning</SelectItem>
                            <SelectItem value="critical">Critical</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="space-y-2">
                        <Label>Notification Channels</Label>
                        <div className="flex flex-wrap gap-2 pt-2">
                          {notificationChannels.map((channel) => {
                            const isSelected = newThreshold.notificationChannels.includes(channel.id)
                            return (
                              <Badge
                                key={channel.id}
                                variant={isSelected ? "default" : "outline"}
                                className="cursor-pointer"
                                onClick={() => {
                                  const updatedChannels = isSelected
                                    ? newThreshold.notificationChannels.filter((c) => c !== channel.id)
                                    : [...newThreshold.notificationChannels, channel.id]
                                  setNewThreshold({ ...newThreshold, notificationChannels: updatedChannels })
                                }}
                              >
                                {channel.name}
                              </Badge>
                            )
                          })}
                        </div>
                      </div>

                      <div className="flex items-center space-x-2 pt-2">
                        <Switch
                          id="threshold-enabled"
                          checked={newThreshold.enabled}
                          onCheckedChange={(checked) => setNewThreshold({ ...newThreshold, enabled: checked })}
                        />
                        <Label htmlFor="threshold-enabled">Enable this threshold</Label>
                      </div>
                    </div>
                  </CardContent>
                  <CardFooter className="flex justify-between">
                    <Button variant="outline" onClick={() => setIsCreating(false)}>
                      Cancel
                    </Button>
                    <Button onClick={handleCreateThreshold}>
                      <Save className="mr-2 h-4 w-4" />
                      Create Threshold
                    </Button>
                  </CardFooter>
                </Card>
              ) : editingThreshold && editingThreshold.metricType === tabValue ? (
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg">Edit Threshold</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-1 gap-4">
                      <div className="space-y-2">
                        <Label htmlFor="edit-threshold-name">Threshold Name</Label>
                        <Input
                          id="edit-threshold-name"
                          value={editingThreshold.name}
                          onChange={(e) => setEditingThreshold({ ...editingThreshold, name: e.target.value })}
                        />
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="edit-metric-name">Metric</Label>
                        <Select
                          value={editingThreshold.metricName}
                          onValueChange={(value) => setEditingThreshold({ ...editingThreshold, metricName: value })}
                        >
                          <SelectTrigger id="edit-metric-name">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {availableMetrics[tabValue as "drift" | "performance"].map((metric) => (
                              <SelectItem key={metric.id} value={metric.id}>
                                {metric.name}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <Label htmlFor="edit-operator">Condition</Label>
                          <Select
                            value={editingThreshold.operator}
                            onValueChange={(value) =>
                              setEditingThreshold({
                                ...editingThreshold,
                                operator: value as ThresholdConfig["operator"],
                              })
                            }
                          >
                            <SelectTrigger id="edit-operator">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value=">">Greater than (&gt;)</SelectItem>
                              <SelectItem value="<">Less than (&lt;)</SelectItem>
                              <SelectItem value=">=">Greater than or equal to (≥)</SelectItem>
                              <SelectItem value="<=">Less than or equal to (≤)</SelectItem>
                              <SelectItem value="=">Equal to (=)</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>

                        <div className="space-y-2">
                          <Label htmlFor="edit-threshold-value">Value</Label>
                          <Input
                            id="edit-threshold-value"
                            type="number"
                            step="0.01"
                            value={editingThreshold.value}
                            onChange={(e) =>
                              setEditingThreshold({
                                ...editingThreshold,
                                value: Number.parseFloat(e.target.value) || 0,
                              })
                            }
                          />
                        </div>
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="edit-severity">Severity</Label>
                        <Select
                          value={editingThreshold.severity}
                          onValueChange={(value) =>
                            setEditingThreshold({
                              ...editingThreshold,
                              severity: value as ThresholdConfig["severity"],
                            })
                          }
                        >
                          <SelectTrigger id="edit-severity">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="info">Info</SelectItem>
                            <SelectItem value="warning">Warning</SelectItem>
                            <SelectItem value="critical">Critical</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="space-y-2">
                        <Label>Notification Channels</Label>
                        <div className="flex flex-wrap gap-2 pt-2">
                          {notificationChannels.map((channel) => {
                            const isSelected = editingThreshold.notificationChannels.includes(channel.id)
                            return (
                              <Badge
                                key={channel.id}
                                variant={isSelected ? "default" : "outline"}
                                className="cursor-pointer"
                                onClick={() => {
                                  const updatedChannels = isSelected
                                    ? editingThreshold.notificationChannels.filter((c) => c !== channel.id)
                                    : [...editingThreshold.notificationChannels, channel.id]
                                  setEditingThreshold({
                                    ...editingThreshold,
                                    notificationChannels: updatedChannels,
                                  })
                                }}
                              >
                                {channel.name}
                              </Badge>
                            )
                          })}
                        </div>
                      </div>

                      <div className="flex items-center space-x-2 pt-2">
                        <Switch
                          id="edit-threshold-enabled"
                          checked={editingThreshold.enabled}
                          onCheckedChange={(checked) => setEditingThreshold({ ...editingThreshold, enabled: checked })}
                        />
                        <Label htmlFor="edit-threshold-enabled">Enable this threshold</Label>
                      </div>
                    </div>
                  </CardContent>
                  <CardFooter className="flex justify-between">
                    <Button variant="outline" onClick={() => setEditingThreshold(null)}>
                      Cancel
                    </Button>
                    <Button onClick={handleUpdateThreshold}>
                      <Save className="mr-2 h-4 w-4" />
                      Update Threshold
                    </Button>
                  </CardFooter>
                </Card>
              ) : (
                <ScrollArea className="h-[400px] rounded-md">
                  <div className="space-y-4 p-1">
                    {thresholds
                      .filter((t) => t.metricType === tabValue)
                      .map((threshold) => (
                        <Card key={threshold.id} className="overflow-hidden">
                          <CardContent className="p-4">
                            <div className="flex items-center justify-between">
                              <div className="flex items-center space-x-2">
                                <Badge className={getSeverityColor(threshold.severity)}>{threshold.severity}</Badge>
                                <h3 className="font-medium">{threshold.name}</h3>
                              </div>
                              <div className="flex items-center space-x-2">
                                <Switch
                                  id={`toggle-${threshold.id}`}
                                  checked={threshold.enabled}
                                  onCheckedChange={(checked) => handleToggleThreshold(threshold.id, checked)}
                                  aria-label={threshold.enabled ? "Disable threshold" : "Enable threshold"}
                                />
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  onClick={() => setEditingThreshold(threshold)}
                                  aria-label="Edit threshold"
                                >
                                  <svg
                                    xmlns="http://www.w3.org/2000/svg"
                                    width="16"
                                    height="16"
                                    viewBox="0 0 24 24"
                                    fill="none"
                                    stroke="currentColor"
                                    strokeWidth="2"
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    className="lucide lucide-pencil"
                                  >
                                    <path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z" />
                                    <path d="m15 5 4 4" />
                                  </svg>
                                </Button>
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  onClick={() => handleDeleteThreshold(threshold.id)}
                                  aria-label="Delete threshold"
                                >
                                  <Trash2 className="h-4 w-4 text-red-500" />
                                </Button>
                              </div>
                            </div>

                            <div className="mt-3 text-sm text-muted-foreground">
                              <p>
                                Alert when <strong>{getMetricName(threshold.metricName)}</strong> is{" "}
                                <strong>{getOperatorLabel(threshold.operator)}</strong>{" "}
                                <strong>{threshold.value}</strong>
                              </p>
                            </div>

                            <div className="mt-3 flex flex-wrap gap-1">
                              {threshold.notificationChannels.map((channel) => (
                                <Badge key={channel} variant="outline" className="text-xs">
                                  {notificationChannels.find((c) => c.id === channel)?.name || channel}
                                </Badge>
                              ))}
                            </div>

                            <div className="mt-3 flex items-center justify-between text-xs text-muted-foreground">
                              <span>
                                Created: {new Date(threshold.createdAt).toLocaleDateString()}{" "}
                                {new Date(threshold.createdAt).toLocaleTimeString()}
                              </span>
                              {threshold.enabled ? (
                                <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                                  <Bell className="mr-1 h-3 w-3" />
                                  Active
                                </Badge>
                              ) : (
                                <Badge variant="outline" className="bg-gray-50 text-gray-500 border-gray-200">
                                  <BellOff className="mr-1 h-3 w-3" />
                                  Disabled
                                </Badge>
                              )}
                            </div>
                          </CardContent>
                        </Card>
                      ))}

                    {thresholds.filter((t) => t.metricType === tabValue).length === 0 && (
                      <div className="flex flex-col items-center justify-center py-8 text-center">
                        <AlertTriangle className="h-12 w-12 text-muted-foreground/50" />
                        <h3 className="mt-4 text-lg font-medium">No thresholds configured</h3>
                        <p className="mt-2 text-sm text-muted-foreground">
                          You haven&apos;t set up any {tabValue} thresholds yet. Click the &quot;New Threshold&quot;
                          button to create one.
                        </p>
                        <Button
                          className="mt-4"
                          onClick={() => {
                            setIsCreating(true)
                            setEditingThreshold(null)
                            setNewThreshold({
                              ...newThresholdTemplate,
                              metricType: tabValue as "drift" | "performance",
                            })
                          }}
                        >
                          <Plus className="mr-2 h-4 w-4" />
                          New Threshold
                        </Button>
                      </div>
                    )}
                  </div>
                </ScrollArea>
              )}
            </TabsContent>
          ))}
        </Tabs>
      </CardContent>
    </Card>
  )
}
