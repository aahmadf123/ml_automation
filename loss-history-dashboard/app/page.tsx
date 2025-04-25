"use client"

import { useState, useEffect } from "react"
import { DashboardHeader } from "@/components/dashboard-header"
import { PageHeader } from "@/components/page-header"
import { QuickStats } from "@/components/quick-stats"
import { PipelineHealth } from "@/components/pipeline-health"
import { ModelPerformanceSummary } from "@/components/model-performance-summary"
import { PerformanceTrends } from "@/components/performance-trends"
import { FeatureDriftSnapshot } from "@/components/feature-drift-snapshot"
import { DataQualityAlerts } from "@/components/data-quality-alerts"
import { RecentMLflowRuns } from "@/components/recent-mlflow-runs"
import { MLQuickActions } from "@/components/ml-quick-actions"
import { MLActivityFeed } from "@/components/ml-activity-feed"
import { DashboardFilter, type FilterOption } from "@/components/dashboard-filter"
import { ActiveFilters } from "@/components/active-filters"
import { DateRangeFilter } from "@/components/date-range-filter"
import { StatusFilter } from "@/components/status-filter"
import { ModelFilter } from "@/components/model-filter"
import { NaturalLanguageQuery } from "@/components/natural-language-query"
import { QueryFeedback } from "@/components/query-feedback"
import { HeroSection } from "@/components/ui/hero-section"
import { ModelComparisonShowcase } from "@/components/ui/model-comparison-showcase"
import { LatestInsightsBanner } from "@/components/ui/latest-insights-banner"
import { GuidedTourButton } from "@/components/guided-tour-button"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { BrainCircuit, ChevronDown, ChevronUp, Filter, PanelRightClose, Bell, AlertTriangle, ArrowUpRight, ArrowDownRight, Zap, Settings, BookMarked, Share2, MessageSquareText, LayoutDashboard, LineChart, Activity, Gauge, Brain } from "lucide-react"
import Link from "next/link"
import type { DateRange } from "react-day-picker"
import { motion } from "framer-motion"
import { MetricCard } from "@/components/ui/metric-card"
import { BusinessImpactProjections } from "@/components/business-impact-projections"
import { PipelineProgress } from "@/components/pipeline-progress"
import { Badge } from "@/components/ui/badge"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { ScrollArea } from "@/components/ui/scroll-area"
import { WhatIfScenarioBuilder } from "@/components/what-if-scenario-builder"

export default function Home() {
  const [mounted, setMounted] = useState(false)
  const [filters, setFilters] = useState<Record<string, any>>({})
  const [dateRange, setDateRange] = useState<DateRange | undefined>({
    from: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
    to: new Date(),
  })
  const [selectedStatuses, setSelectedStatuses] = useState<string[]>([])
  const [selectedModels, setSelectedModels] = useState<string[]>([])
  const [lastQuery, setLastQuery] = useState("")
  const [showFeedback, setShowFeedback] = useState(false)
  const [sortConfig, setSortConfig] = useState<{ field: string; direction: "asc" | "desc" } | null>(null)
  const [showFilters, setShowFilters] = useState(false)
  const [showAdvancedContent, setShowAdvancedContent] = useState(false)
  const [activeTab, setActiveTab] = useState("performance")
  const [pinnedWidgets, setPinnedWidgets] = useState<string[]>(["quickStats", "pipelineProgress", "modelComparison"])
  const [hasAlerts, setHasAlerts] = useState(true)
  const [dashboardView, setDashboardView] = useState("default")

  // Sample critical alerts
  const criticalAlerts = [
    { id: 1, type: "drift", severity: "critical", message: "Feature drift detected in 'customer_age' beyond threshold", time: "10 min ago" },
    { id: 2, type: "performance", severity: "warning", message: "Model accuracy dropped by 3.2% in last 24 hours", time: "2 hours ago" },
    { id: 3, type: "infrastructure", severity: "info", message: "Scheduled retraining completed successfully", time: "6 hours ago" },
  ]

  // Sample insights
  const aiInsights = [
    { id: 1, insight: "Recent feature importance shift suggests revisiting the 'payment_history' feature engineering", impact: "high" },
    { id: 2, insight: "Consider adjusting the learning rate to improve convergence speed", impact: "medium" },
    { id: 3, insight: "Data distribution in 'transaction_amount' shows potential outliers affecting model performance", impact: "high" },
  ]

  // Sample annotations
  const recentAnnotations = [
    { id: 1, user: "Alex Kim", text: "Investigated the performance drop - related to new data source integration", time: "Yesterday", avatar: "/avatars/alex.png" },
    { id: 2, user: "Jamie Chen", text: "Added new features to improve prediction accuracy", time: "2 days ago", avatar: "/avatars/jamie.png" },
  ]

  const filterOptions: FilterOption[] = [
    {
      id: "model",
      label: "Model",
      type: "multiselect",
      options: [
        { value: "model1", label: "Traditional Model (48 Attributes)" },
        { value: "model4", label: "Enhanced Model (Fast Decay)" },
      ],
    },
    {
      id: "status",
      label: "Status",
      type: "multiselect",
      options: [
        { value: "success", label: "Success" },
        { value: "failed", label: "Failed" },
        { value: "warning", label: "Warning" },
        { value: "running", label: "Running" },
      ],
    },
    {
      id: "severity",
      label: "Severity",
      type: "select",
      options: [
        { value: "critical", label: "Critical" },
        { value: "warning", label: "Warning" },
        { value: "info", label: "Info" },
      ],
    },
    {
      id: "date",
      label: "Date Range",
      type: "daterange",
    },
    {
      id: "search",
      label: "Search",
      type: "search",
    },
  ]

  const filterLabels: Record<string, string> = {
    model: "Model",
    status: "Status",
    severity: "Severity",
    date: "Date Range",
    search: "Search",
    metric: "Metric Filter",
  }

  const handleFilterChange = (newFilters: Record<string, any>) => {
    // Extract sorting information if present
    if (newFilters._sort) {
      setSortConfig(newFilters._sort)
      delete newFilters._sort
    }

    // Store the original query if provided
    const query = newFilters._originalQuery || ""
    if (query) {
      setLastQuery(query)
      setShowFeedback(true)
      delete newFilters._originalQuery
    }

    // Remove other internal properties
    delete newFilters._intent

    setFilters(newFilters)

    // Update other filter states based on the new filters
    if (newFilters.date) {
      setDateRange(newFilters.date)
    }

    if (newFilters.status) {
      setSelectedStatuses(Array.isArray(newFilters.status) ? newFilters.status : [newFilters.status])
    } else {
      setSelectedStatuses([])
    }

    if (newFilters.model) {
      setSelectedModels(Array.isArray(newFilters.model) ? newFilters.model : [newFilters.model])
    } else {
      setSelectedModels([])
    }

    console.log("Filters changed:", newFilters)
    if (sortConfig) {
      console.log("Sorting by:", sortConfig.field, sortConfig.direction)
    }
  }

  const handleRemoveFilter = (key: string) => {
    const newFilters = { ...filters }
    delete newFilters[key]
    setFilters(newFilters)

    // Update other filter states
    if (key === "date") {
      setDateRange({
        from: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
        to: new Date(),
      })
    } else if (key === "status") {
      setSelectedStatuses([])
    } else if (key === "model") {
      setSelectedModels([])
    }
  }

  const clearAllFilters = () => {
    setFilters({})
    setDateRange({
      from: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
      to: new Date(),
    })
    setSelectedStatuses([])
    setSelectedModels([])
    setLastQuery("")
    setShowFeedback(false)
    setSortConfig(null)
  }

  const handleDateRangeChange = (range: DateRange | undefined) => {
    setDateRange(range)
    const newFilters = { ...filters, date: range }
    setFilters(newFilters)
  }

  const handleStatusChange = (statuses: string[]) => {
    setSelectedStatuses(statuses)
    const newFilters = { ...filters, status: statuses }
    if (statuses.length === 0) {
      delete newFilters.status
    }
    setFilters(newFilters)
  }

  const handleModelChange = (models: string[]) => {
    setSelectedModels(models)
    const newFilters = { ...filters, model: models }
    if (models.length === 0) {
      delete newFilters.model
    }
    setFilters(newFilters)
  }

  const handleRephrase = (newQuery: string) => {
    // Handle when a user clicks on a suggested query rephrasing
    setLastQuery(newQuery)
    setShowFeedback(false)

    // Simulate the user entering and executing this query
    const inputElement = document.querySelector('input[placeholder*="Ask anything"]') as HTMLInputElement
    if (inputElement) {
      inputElement.value = newQuery
      // Dispatch events to simulate user input
      const inputEvent = new Event("input", { bubbles: true })
      inputElement.dispatchEvent(inputEvent)

      // Submit the query
      setTimeout(() => {
        const searchButton = document.querySelector('button:contains("Search")') as HTMLButtonElement
        if (searchButton) {
          searchButton.click()
        }
      }, 100)
    }
  }

  const togglePinWidget = (widgetId: string) => {
    setPinnedWidgets(current => 
      current.includes(widgetId) 
        ? current.filter(id => id !== widgetId) 
        : [...current, widgetId]
    )
  }

  const dismissAlert = (alertId: number) => {
    if (criticalAlerts.length <= 1) {
      setHasAlerts(false)
    }
  }

  const changeDashboardView = (viewType: string) => {
    setDashboardView(viewType)
  }

  // Sample model data for the filter
  const models = [
    { id: "model1", name: "Traditional Model (48 Attributes)", type: "Regression" },
    { id: "model4", name: "Enhanced Model (Fast Decay)", type: "Regression" },
  ]

  useEffect(() => {
    // Add a subtle background pattern animation
    const body = document.querySelector('body')
    if (body) {
      body.classList.add('animated-bg-pattern')
    }
    
    return () => {
      if (body) {
        body.classList.remove('animated-bg-pattern')
      }
    }
  }, [])

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) return null

  return (
    <div className="flex flex-col min-h-screen">
      <DashboardHeader />
      <div className="flex-1 p-4 space-y-6 md:p-6 mx-auto w-full max-w-7xl">
        {/* Summary/Header Section */}
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between mb-4">
          <div>
            <h1 className="text-2xl font-bold flex items-center">
              Model Performance & Health
              {hasAlerts && (
                <Badge variant="destructive" className="ml-2 animate-pulse">
                  <AlertTriangle className="h-3 w-3 mr-1" />
                  Alerts
                </Badge>
              )}
            </h1>
            <p className="text-muted-foreground">
              Monitor model performance, health and impact with interactive analytics
            </p>
          </div>
          
          <div className="flex items-center gap-3">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="outline" size="sm" className="gap-1">
                    <Share2 className="h-4 w-4" />
                    Share
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  Share this dashboard
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
            
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="outline" size="sm" className="gap-1">
                    <BookMarked className="h-4 w-4" />
                    Save View
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  Save this dashboard configuration
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
            
            <div className="flex items-center gap-1 border px-2 py-1 rounded-md">
              <span className="text-sm font-medium mr-1">View:</span>
              <Button 
                variant={dashboardView === "default" ? "secondary" : "ghost"} 
                size="sm" 
                className="h-7 px-2"
                onClick={() => changeDashboardView("default")}
              >
                <LayoutDashboard className="h-4 w-4" />
              </Button>
              <Button 
                variant={dashboardView === "technical" ? "secondary" : "ghost"} 
                size="sm" 
                className="h-7 px-2"
                onClick={() => changeDashboardView("technical")}
              >
                <LineChart className="h-4 w-4" />
              </Button>
              <Button 
                variant={dashboardView === "executive" ? "secondary" : "ghost"} 
                size="sm" 
                className="h-7 px-2"
                onClick={() => changeDashboardView("executive")}
              >
                <Activity className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
        
        {/* Critical Alerts Section */}
        {hasAlerts && (
          <Card className="border-red-200 bg-red-50 dark:bg-red-950/20 dark:border-red-900 mb-4">
            <CardHeader className="py-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-red-800 dark:text-red-300 flex items-center gap-2">
                  <AlertTriangle className="h-5 w-5" />
                  Critical Alerts
                </CardTitle>
                <Button variant="outline" size="sm" className="h-7" onClick={() => setHasAlerts(false)}>
                  Dismiss All
                </Button>
              </div>
            </CardHeader>
            <CardContent className="py-0">
              <div className="space-y-2">
                {criticalAlerts.map(alert => (
                  <div key={alert.id} className="flex items-center justify-between bg-white dark:bg-gray-800 p-3 rounded-md border border-red-100 dark:border-red-900/50">
                    <div className="flex items-center gap-3">
                      {alert.severity === "critical" && <span className="h-2 w-2 rounded-full bg-red-500"></span>}
                      {alert.severity === "warning" && <span className="h-2 w-2 rounded-full bg-yellow-500"></span>}
                      {alert.severity === "info" && <span className="h-2 w-2 rounded-full bg-blue-500"></span>}
                      <div>
                        <p className="font-medium">{alert.message}</p>
                        <p className="text-xs text-gray-500">{alert.time}</p>
                      </div>
                    </div>
                    <Button variant="ghost" size="sm" className="h-7" onClick={() => dismissAlert(alert.id)}>
                      View Details
                    </Button>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
        
        {/* Global Filters */}
        <Card className="shadow-sm border-t-4 border-t-blue-500">
          <CardContent className="pt-6">
            <div className="w-full mb-4">
              <NaturalLanguageQuery onFilterChange={handleFilterChange} />

              {showFeedback && (
                <QueryFeedback
                  query={lastQuery}
                  parsedFilters={filters}
                  onDismiss={() => setShowFeedback(false)}
                  onRephrase={handleRephrase}
                />
              )}
            </div>
            
            <div className="flex justify-center">
              <Button 
                variant="outline" 
                size="sm" 
                onClick={() => setShowFilters(!showFilters)}
                className="flex items-center gap-1"
              >
                <Filter className="h-4 w-4" />
                {showFilters ? "Hide Filters" : "Show Filters"}
                {showFilters ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
              </Button>
            </div>
          </CardContent>
        </Card>
        
        {/* Filters Section (Collapsible) */}
        {showFilters && (
          <div className="space-y-4 bg-slate-50 dark:bg-slate-900 p-4 rounded-lg border animate-slideDown">
            <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
              <DashboardFilter
                filterOptions={filterOptions}
                onFilterChange={handleFilterChange}
                className="w-full md:w-auto"
              />

              <ModelFilter
                models={models}
                selectedModels={selectedModels}
                onSelectionChange={handleModelChange}
                className="w-full md:w-auto"
              />
            </div>

            <div className="space-y-4">
              <DateRangeFilter onDateRangeChange={handleDateRangeChange} className="w-full" />

              <StatusFilter onStatusChange={handleStatusChange} className="w-full" />

              <ActiveFilters
                filters={filters}
                filterLabels={filterLabels}
                onRemoveFilter={handleRemoveFilter}
                onClearAll={clearAllFilters}
                className="w-full"
              />
            </div>
          </div>
        )}
        
        {/* Dashboard Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="performance" className="flex items-center gap-1">
              <Gauge className="h-4 w-4" />
              Performance
            </TabsTrigger>
            <TabsTrigger value="health" className="flex items-center gap-1">
              <Activity className="h-4 w-4" />
              Health
            </TabsTrigger>
            <TabsTrigger value="impact" className="flex items-center gap-1">
              <Brain className="h-4 w-4" />
              Impact
            </TabsTrigger>
          </TabsList>
          
          {/* Performance Tab Content */}
          <TabsContent value="performance" className="space-y-4 mt-4">
            {/* Key Performance Indicators */}
            <Card className="shadow-sm">
              <CardHeader className="pb-2">
                <CardTitle className="text-lg flex items-center justify-between">
                  <span>Key Performance Metrics</span>
                  <Button variant="ghost" size="sm" onClick={() => togglePinWidget('quickStats')}>
                    {pinnedWidgets.includes('quickStats') ? 'Unpin' : 'Pin'}
                  </Button>
                </CardTitle>
                <CardDescription>Overall model performance at a glance</CardDescription>
              </CardHeader>
              <CardContent>
                <QuickStats />
              </CardContent>
            </Card>
            
            {/* Model Comparison */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <Card className="shadow-sm">
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">Traditional Model</CardTitle>
                  <CardDescription>48 Attributes Model Performance</CardDescription>
                </CardHeader>
                <CardContent className="pb-2">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="text-2xl font-bold">67%</h3>
                    <Badge variant="outline" className="bg-blue-50 text-blue-800">R² Score</Badge>
                  </div>
                  
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="text-sm font-medium">Accuracy</span>
                        <span className="text-sm font-medium">72.4%</span>
                      </div>
                      <div className="h-2 w-full bg-blue-100 rounded-full">
                        <div className="h-full bg-blue-500 rounded-full" style={{ width: '72.4%' }}></div>
                      </div>
                    </div>
                    
                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="text-sm font-medium">Precision</span>
                        <span className="text-sm font-medium">68.9%</span>
                      </div>
                      <div className="h-2 w-full bg-blue-100 rounded-full">
                        <div className="h-full bg-blue-500 rounded-full" style={{ width: '68.9%' }}></div>
                      </div>
                    </div>
                    
                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="text-sm font-medium">Recall</span>
                        <span className="text-sm font-medium">70.2%</span>
                      </div>
                      <div className="h-2 w-full bg-blue-100 rounded-full">
                        <div className="h-full bg-blue-500 rounded-full" style={{ width: '70.2%' }}></div>
                      </div>
                    </div>
                  </div>
                </CardContent>
                <CardFooter className="pt-0">
                  <Button variant="ghost" size="sm" className="w-full justify-start text-blue-600">
                    View detailed metrics
                  </Button>
                </CardFooter>
              </Card>
              
              <Card className="shadow-sm">
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">Enhanced Model</CardTitle>
                  <CardDescription>Fast Decay Model Performance</CardDescription>
                </CardHeader>
                <CardContent className="pb-2">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="text-2xl font-bold">79%</h3>
                    <Badge variant="outline" className="bg-green-50 text-green-800">R² Score</Badge>
                  </div>
                  
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="text-sm font-medium">Accuracy</span>
                        <span className="text-sm font-medium flex items-center">
                          83.7%
                          <ArrowUpRight className="h-3 w-3 text-green-500 ml-1" />
                        </span>
                      </div>
                      <div className="h-2 w-full bg-green-100 rounded-full">
                        <div className="h-full bg-green-500 rounded-full" style={{ width: '83.7%' }}></div>
                      </div>
                    </div>
                    
                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="text-sm font-medium">Precision</span>
                        <span className="text-sm font-medium flex items-center">
                          81.2%
                          <ArrowUpRight className="h-3 w-3 text-green-500 ml-1" />
                        </span>
                      </div>
                      <div className="h-2 w-full bg-green-100 rounded-full">
                        <div className="h-full bg-green-500 rounded-full" style={{ width: '81.2%' }}></div>
                      </div>
                    </div>
                    
                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="text-sm font-medium">Recall</span>
                        <span className="text-sm font-medium flex items-center">
                          80.5%
                          <ArrowUpRight className="h-3 w-3 text-green-500 ml-1" />
                        </span>
                      </div>
                      <div className="h-2 w-full bg-green-100 rounded-full">
                        <div className="h-full bg-green-500 rounded-full" style={{ width: '80.5%' }}></div>
                      </div>
                    </div>
                  </div>
                </CardContent>
                <CardFooter className="pt-0">
                  <Button variant="ghost" size="sm" className="w-full justify-start text-green-600">
                    View detailed metrics
                  </Button>
                </CardFooter>
              </Card>
            </div>
            
            {/* Pipeline Progress (Interactive) */}
            <Card className="shadow-sm">
              <CardHeader className="pb-2">
                <CardTitle className="text-lg flex items-center justify-between">
                  <span>Pipeline Progress</span>
                  <Button variant="ghost" size="sm" onClick={() => togglePinWidget('pipelineProgress')}>
                    {pinnedWidgets.includes('pipelineProgress') ? 'Unpin' : 'Pin'}
                  </Button>
                </CardTitle>
                <CardDescription>Click on stages for detailed metrics</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="max-w-4xl mx-auto">
                  <PipelineProgress />
                </div>
              </CardContent>
            </Card>
            
            {/* Performance Trends */}
            <Card className="shadow-sm">
              <CardHeader className="pb-2">
                <CardTitle className="text-lg">Performance Over Time</CardTitle>
                <CardDescription>Key metrics tracked over the selected time period</CardDescription>
              </CardHeader>
              <CardContent>
                <PerformanceTrends />
              </CardContent>
            </Card>
          </TabsContent>
          
          {/* Health Tab Content */}
          <TabsContent value="health" className="space-y-4 mt-4">
            {/* Data Quality */}
            <Card className="shadow-sm">
              <CardHeader className="pb-2">
                <CardTitle className="text-lg">Data Quality Health</CardTitle>
                <CardDescription>Metrics on data completeness, consistency, and accuracy</CardDescription>
              </CardHeader>
              <CardContent>
                <DataQualityAlerts />
              </CardContent>
            </Card>
            
            {/* Feature Drift */}
            <Card className="shadow-sm">
              <CardHeader className="pb-2">
                <CardTitle className="text-lg">Feature Drift Detection</CardTitle>
                <CardDescription>Monitoring changes in feature distributions</CardDescription>
              </CardHeader>
              <CardContent>
                <FeatureDriftSnapshot />
              </CardContent>
            </Card>
            
            {/* Pipeline Health */}
            <Card className="shadow-sm">
              <CardHeader className="pb-2">
                <CardTitle className="text-lg">Pipeline Health</CardTitle>
                <CardDescription>Overall system health and performance</CardDescription>
              </CardHeader>
              <CardContent>
                <PipelineHealth />
              </CardContent>
            </Card>
            
            {/* Recent ML Flow Runs */}
            <Card className="shadow-sm">
              <CardHeader className="pb-2">
                <CardTitle className="text-lg">Recent MLflow Runs</CardTitle>
                <CardDescription>Latest experiment runs and results</CardDescription>
              </CardHeader>
              <CardContent>
                <RecentMLflowRuns />
              </CardContent>
            </Card>
          </TabsContent>
          
          {/* Impact Tab Content */}
          <TabsContent value="impact" className="space-y-4 mt-4">
            {/* What-If Scenario Builder */}
            <WhatIfScenarioBuilder />
            
            {/* Business Impact */}
            <Card className="shadow-sm">
              <CardHeader className="pb-2">
                <CardTitle className="text-lg">Business Impact</CardTitle>
                <CardDescription>ROI and business value metrics</CardDescription>
              </CardHeader>
              <CardContent>
                <BusinessImpactProjections />
              </CardContent>
            </Card>
            
            {/* Before and After Results */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="shadow-lg overflow-hidden border-0 bg-gradient-to-br from-red-50 to-red-100 dark:from-red-950 dark:to-red-900">
                <CardContent className="p-6">
                  <h2 className="text-2xl font-bold mb-4 text-red-800 dark:text-red-300">Before ML Automation</h2>
                  <div className="space-y-4">
                    <div className="flex justify-between items-center">
                      <span className="font-medium">Processing Time:</span>
                      <span className="font-bold text-red-700 dark:text-red-400">5.7 days</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="font-medium">Accuracy:</span>
                      <span className="font-bold text-red-700 dark:text-red-400">87.6%</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="font-medium">Manual Reviews:</span>
                      <span className="font-bold text-red-700 dark:text-red-400">42%</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="font-medium">False Positives:</span>
                      <span className="font-bold text-red-700 dark:text-red-400">13.2%</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
              
              <Card className="shadow-lg overflow-hidden border-0 bg-gradient-to-br from-green-50 to-green-100 dark:from-green-950 dark:to-green-900">
                <CardContent className="p-6">
                  <h2 className="text-2xl font-bold mb-4 text-green-800 dark:text-green-300">After ML Automation</h2>
                  <div className="space-y-4">
                    <div className="flex justify-between items-center">
                      <span className="font-medium">Processing Time:</span>
                      <span className="font-bold text-green-700 dark:text-green-400">3.2 days</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="font-medium">Accuracy:</span>
                      <span className="font-bold text-green-700 dark:text-green-400">94.2%</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="font-medium">Manual Reviews:</span>
                      <span className="font-bold text-green-700 dark:text-green-400">18%</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="font-medium">False Positives:</span>
                      <span className="font-bold text-green-700 dark:text-green-400">6.7%</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
            
            {/* Model Comparison Showcase */}
            <Card className="shadow-sm">
              <CardHeader className="pb-2">
                <CardTitle className="text-lg">Model Comparison Impact</CardTitle>
                <CardDescription>Comparative analysis of model performance</CardDescription>
              </CardHeader>
              <CardContent>
                <ModelComparisonShowcase />
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
        
        {/* AI Insights & Collaboration Section */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* AI-Generated Insights */}
          <Card className="shadow-sm lg:col-span-2">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg flex items-center gap-2">
                <Zap className="h-5 w-5 text-amber-500" />
                AI-Generated Insights
              </CardTitle>
              <CardDescription>Automatically generated recommendations based on current data</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {aiInsights.map(insight => (
                  <div key={insight.id} className="p-3 border rounded-md bg-amber-50 dark:bg-amber-950/20">
                    <div className="flex items-start gap-2">
                      <Badge variant="outline" className={
                        insight.impact === "high" ? "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300" :
                        insight.impact === "medium" ? "bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300" :
                        "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300"
                      }>
                        {insight.impact} impact
                      </Badge>
                      <p className="flex-1">{insight.insight}</p>
                    </div>
                    <div className="flex justify-end mt-2">
                      <Button variant="ghost" size="sm" className="h-7 text-xs">Implement</Button>
                      <Button variant="ghost" size="sm" className="h-7 text-xs">Dismiss</Button>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
            <CardFooter>
              <Button variant="outline" size="sm" className="w-full">
                <BrainCircuit className="h-4 w-4 mr-2" />
                Generate More Insights
              </Button>
            </CardFooter>
          </Card>
          
          {/* Collaboration & Annotations */}
          <Card className="shadow-sm">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg flex items-center gap-2">
                <MessageSquareText className="h-5 w-5 text-blue-500" />
                Team Annotations
              </CardTitle>
              <CardDescription>Comments and notes from your team</CardDescription>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[220px] pr-4">
                <div className="space-y-4">
                  {recentAnnotations.map(annotation => (
                    <div key={annotation.id} className="flex gap-3 pb-4 border-b last:border-0">
                      <Avatar className="h-8 w-8">
                        <AvatarFallback>{annotation.user.charAt(0)}</AvatarFallback>
                      </Avatar>
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-sm">{annotation.user}</span>
                          <span className="text-xs text-gray-500">{annotation.time}</span>
                        </div>
                        <p className="text-sm">{annotation.text}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
            <CardFooter>
              <Button variant="outline" size="sm" className="w-full">
                Add Annotation
              </Button>
            </CardFooter>
          </Card>
        </div>
        
        {/* Actions Section */}
        <Card className="shadow-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-lg">Quick Actions</CardTitle>
            <CardDescription>Common operations and tools</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <Button variant="outline" className="h-auto py-4 flex flex-col items-center justify-center">
                <Activity className="h-6 w-6 mb-2" />
                <span>Run Performance Check</span>
              </Button>
              <Button variant="outline" className="h-auto py-4 flex flex-col items-center justify-center">
                <Zap className="h-6 w-6 mb-2" />
                <span>Schedule Retraining</span>
              </Button>
              <Button variant="outline" className="h-auto py-4 flex flex-col items-center justify-center">
                <Settings className="h-6 w-6 mb-2" />
                <span>Configure Thresholds</span>
              </Button>
              <Button variant="outline" className="h-auto py-4 flex flex-col items-center justify-center">
                <Share2 className="h-6 w-6 mb-2" />
                <span>Export Report</span>
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
