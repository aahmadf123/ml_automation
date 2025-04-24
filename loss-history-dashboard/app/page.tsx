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
import { Card, CardContent } from "@/components/ui/card"
import { BrainCircuit, ChevronDown, ChevronUp, Filter, PanelRightClose } from "lucide-react"
import Link from "next/link"
import type { DateRange } from "react-day-picker"
import { motion } from "framer-motion"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { MetricCard } from "@/components/ui/metric-card"

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

  const filterOptions: FilterOption[] = [
    {
      id: "model",
      label: "Model",
      type: "multiselect",
      options: [
        { value: "model1", label: "Gradient Boosting (model1)" },
        { value: "model2", label: "Random Forest (model2)" },
        { value: "model3", label: "XGBoost (model3)" },
        { value: "model4", label: "Neural Network (model4)" },
        { value: "model5", label: "LightGBM (model5)" },
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

  // Sample model data for the filter
  const models = [
    { id: "model1", name: "Gradient Boosting (model1)", type: "Regression" },
    { id: "model2", name: "Random Forest (model2)", type: "Regression" },
    { id: "model3", name: "XGBoost (model3)", type: "Regression" },
    { id: "model4", name: "Neural Network (model4)", type: "Classification" },
    { id: "model5", name: "LightGBM (model5)", type: "Classification" },
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
      <div className="flex-1 p-4 space-y-6 md:p-6 lg:ml-64">
        {/* Hero Section */}
        <HeroSection />
        
        {/* Latest Insights Banner */}
        <LatestInsightsBanner />
        
        {/* AI Assistant & Guided Tour Buttons */}
        <div className="flex flex-wrap gap-4 justify-center mb-6">
          <Link href="/model-comparison/ai-assistant">
            <Button size="lg" className="bg-gradient-to-r from-blue-500 to-indigo-600 text-white hover:from-blue-600 hover:to-indigo-700 gap-2 shadow-lg transform transition-transform hover:scale-105">
              <BrainCircuit className="h-5 w-5" />
              Try AI Assistant Demo
            </Button>
          </Link>
          
          <GuidedTourButton />
        </div>
        
        {/* Enhanced Quick Stats */}
        <QuickStats />
        
        {/* Model Comparison Showcase */}
        <ModelComparisonShowcase />

        {/* Natural Language Query Section */}
        <Card className="shadow-lg border-t-4 border-t-blue-500">
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
        
        {/* Toggle Advanced Content Button */}
        <div className="flex justify-center my-6">
          <Button 
            variant="outline" 
            onClick={() => setShowAdvancedContent(!showAdvancedContent)}
            className="flex items-center gap-1"
          >
            {showAdvancedContent ? "Hide Advanced Details" : "Show Advanced Details"}
            {showAdvancedContent ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
          </Button>
        </div>

        {/* Advanced Content (Collapsible) */}
        {showAdvancedContent && (
          <div className="space-y-6 animate-fadeIn">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <PipelineHealth />
              <ModelPerformanceSummary />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <PerformanceTrends />
              <FeatureDriftSnapshot />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <DataQualityAlerts />
              <RecentMLflowRuns />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <MLQuickActions />
              <MLActivityFeed />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
