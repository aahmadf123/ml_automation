"use client"

import { useEffect, useState } from "react"
import { Download, Play, RefreshCw, BarChart, LineChart, Eye, Search, Filter } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"

interface MLflowRun {
  runId: string
  modelId: string
  rmse: number
  timestamp: string
  status: "FINISHED" | "RUNNING" | "FAILED"
  duration: number
  parameters?: Record<string, any>
  metrics?: Record<string, number>
  tags?: Record<string, string>
  artifacts?: { name: string; type: string; size: string; lastModified: string }[]
  logs?: string
}

export function RecentMLflowRuns() {
  const [loading, setLoading] = useState(true)
  const [runs, setRuns] = useState<MLflowRun[]>([])
  const [filteredRuns, setFilteredRuns] = useState<MLflowRun[]>([])
  const [selectedRun, setSelectedRun] = useState<MLflowRun | null>(null)
  const [activeTab, setActiveTab] = useState("overview")

  // Filter states
  const [searchQuery, setSearchQuery] = useState("")
  const [statusFilter, setStatusFilter] = useState<string>("all")
  const [modelFilter, setModelFilter] = useState<string[]>([])
  const [sortBy, setSortBy] = useState<string>("timestamp")
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc")

  useEffect(() => {
    // Simulate API call to MLflow
    const fetchRecentRuns = async () => {
      // In a real implementation, this would be an API call to our internal API that fetches from MLflow
      // No external links needed as everything is integrated

      // Simulated data
      setTimeout(() => {
        const mockData: MLflowRun[] = [
          {
            runId: "run_123456",
            modelId: "model3",
            rmse: 0.0632,
            timestamp: "2023-04-15T08:15:33",
            status: "FINISHED",
            duration: 3245,
            parameters: {
              learning_rate: 0.05,
              max_depth: 7,
              n_estimators: 300,
              subsample: 0.85,
              colsample_bytree: 0.75,
            },
            metrics: {
              rmse: 0.0632,
              mse: 0.004,
              mae: 0.0487,
              r2: 0.9512,
            },
            tags: {
              author: "john.doe",
              experiment: "xgboost_optimization",
              version: "7",
            },
            artifacts: [
              { name: "model.pkl", type: "Model", size: "48.2 MB", lastModified: "2023-04-15T08:15:33" },
              { name: "requirements.txt", type: "Text", size: "1.3 KB", lastModified: "2023-04-15T08:15:33" },
              { name: "learning_curve.png", type: "Image", size: "178 KB", lastModified: "2023-04-15T08:15:33" },
              { name: "metrics.json", type: "JSON", size: "4.8 KB", lastModified: "2023-04-15T08:15:33" },
            ],
            logs: "INFO - Starting model training\nINFO - Loading data\nINFO - Preprocessing data\nINFO - Training model with parameters: learning_rate=0.05, max_depth=7, n_estimators=300\nINFO - Training completed in 3245 seconds\nINFO - Model evaluation: RMSE=0.0632, MSE=0.004, MAE=0.0487, R2=0.9512\nINFO - Saving model artifacts\nINFO - Run completed successfully",
          },
          {
            runId: "run_123455",
            modelId: "model1",
            rmse: 0.0842,
            timestamp: "2023-04-15T07:45:12",
            status: "FINISHED",
            duration: 2876,
            parameters: {
              learning_rate: 0.1,
              max_depth: 6,
              n_estimators: 200,
              subsample: 0.8,
              colsample_bytree: 0.8,
            },
            metrics: {
              rmse: 0.0842,
              mse: 0.0071,
              mae: 0.0623,
              r2: 0.9231,
            },
            tags: {
              author: "jane.smith",
              experiment: "gradient_boosting_baseline",
              version: "5",
            },
            artifacts: [
              { name: "model.pkl", type: "Model", size: "42.3 MB", lastModified: "2023-04-15T07:45:12" },
              { name: "requirements.txt", type: "Text", size: "1.2 KB", lastModified: "2023-04-15T07:45:12" },
              { name: "feature_importance.png", type: "Image", size: "156 KB", lastModified: "2023-04-15T07:45:12" },
              { name: "metrics.json", type: "JSON", size: "4.5 KB", lastModified: "2023-04-15T07:45:12" },
            ],
            logs: "INFO - Starting model training\nINFO - Loading data\nINFO - Preprocessing data\nINFO - Training model with parameters: learning_rate=0.1, max_depth=6, n_estimators=200\nINFO - Training completed in 2876 seconds\nINFO - Model evaluation: RMSE=0.0842, MSE=0.0071, MAE=0.0623, R2=0.9231\nINFO - Saving model artifacts\nINFO - Run completed successfully",
          },
          {
            runId: "run_123454",
            modelId: "model5",
            rmse: 0.0921,
            timestamp: "2023-04-14T22:30:45",
            status: "FINISHED",
            duration: 3102,
            parameters: {
              learning_rate: 0.08,
              max_depth: 5,
              n_estimators: 250,
              subsample: 0.8,
              colsample_bytree: 0.8,
            },
            metrics: {
              rmse: 0.0921,
              mse: 0.0085,
              mae: 0.0714,
              r2: 0.9102,
            },
            tags: {
              author: "john.doe",
              experiment: "lightgbm_categorical",
              version: "4",
            },
            artifacts: [
              { name: "model.pkl", type: "Model", size: "38.9 MB", lastModified: "2023-04-14T22:30:45" },
              { name: "requirements.txt", type: "Text", size: "1.2 KB", lastModified: "2023-04-14T22:30:45" },
              { name: "feature_importance.png", type: "Image", size: "162 KB", lastModified: "2023-04-14T22:30:45" },
              { name: "metrics.json", type: "JSON", size: "4.6 KB", lastModified: "2023-04-14T22:30:45" },
            ],
            logs: "INFO - Starting model training\nINFO - Loading data\nINFO - Preprocessing data\nINFO - Training model with parameters: learning_rate=0.08, max_depth=5, n_estimators=250\nINFO - Training completed in 3102 seconds\nINFO - Model evaluation: RMSE=0.0921, MSE=0.0085, MAE=0.0714, R2=0.9102\nINFO - Saving model artifacts\nINFO - Run completed successfully",
          },
          {
            runId: "run_123453",
            modelId: "model2",
            rmse: 0.1245,
            timestamp: "2023-04-14T18:12:33",
            status: "FAILED",
            duration: 1523,
            parameters: {
              n_estimators: 150,
              max_depth: 8,
              min_samples_split: 5,
              min_samples_leaf: 2,
              bootstrap: true,
            },
            metrics: {},
            tags: {
              author: "jane.smith",
              experiment: "random_forest_optimization",
              version: "3",
            },
            artifacts: [{ name: "error_log.txt", type: "Text", size: "8.5 KB", lastModified: "2023-04-14T18:12:33" }],
            logs: "INFO - Starting model training\nINFO - Loading data\nINFO - Preprocessing data\nINFO - Training model with parameters: n_estimators=150, max_depth=8\nERROR - Out of memory error during training\nERROR - Run failed with error: MemoryError",
          },
          {
            runId: "run_123452",
            modelId: "model4",
            rmse: 0.1532,
            timestamp: "2023-04-14T15:05:21",
            status: "FINISHED",
            duration: 4256,
            parameters: {
              hidden_layer_sizes: [128, 64, 32],
              activation: "relu",
              solver: "adam",
              alpha: 0.0001,
              batch_size: 64,
            },
            metrics: {
              rmse: 0.1532,
              mse: 0.0235,
              mae: 0.1124,
              r2: 0.8321,
            },
            tags: {
              author: "john.doe",
              experiment: "neural_network",
              version: "2",
            },
            artifacts: [
              { name: "model.h5", type: "Model", size: "56.4 MB", lastModified: "2023-04-14T15:05:21" },
              { name: "requirements.txt", type: "Text", size: "2.1 KB", lastModified: "2023-04-14T15:05:21" },
              { name: "training_history.png", type: "Image", size: "185 KB", lastModified: "2023-04-14T15:05:21" },
              { name: "metrics.json", type: "JSON", size: "5.2 KB", lastModified: "2023-04-14T15:05:21" },
            ],
            logs: "INFO - Starting model training\nINFO - Loading data\nINFO - Preprocessing data\nINFO - Training neural network with parameters: hidden_layer_sizes=[128, 64, 32], activation=relu\nINFO - Training completed in 4256 seconds\nINFO - Model evaluation: RMSE=0.1532, MSE=0.0235, MAE=0.1124, R2=0.8321\nINFO - Saving model artifacts\nINFO - Run completed successfully",
          },
          {
            runId: "run_123451",
            modelId: "model3",
            rmse: 0.0712,
            timestamp: "2023-04-14T12:30:15",
            status: "RUNNING",
            duration: 1800, // Current duration
            parameters: {
              learning_rate: 0.05,
              max_depth: 7,
              n_estimators: 300,
              subsample: 0.85,
              colsample_bytree: 0.75,
            },
            metrics: {},
            tags: {
              author: "john.doe",
              experiment: "xgboost_optimization",
              version: "6",
            },
            artifacts: [],
            logs: "INFO - Starting model training\nINFO - Loading data\nINFO - Preprocessing data\nINFO - Training model with parameters: learning_rate=0.05, max_depth=7, n_estimators=300\nINFO - Training in progress...",
          },
        ]

        setRuns(mockData)
        setFilteredRuns(mockData)
        setLoading(false)
      }, 1400)
    }

    fetchRecentRuns()
  }, [])

  // Apply filters whenever filter states change
  useEffect(() => {
    let result = [...runs]

    // Apply search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      result = result.filter(
        (run) =>
          run.runId.toLowerCase().includes(query) ||
          run.modelId.toLowerCase().includes(query) ||
          (run.tags?.author && run.tags.author.toLowerCase().includes(query)),
      )
    }

    // Apply status filter
    if (statusFilter !== "all") {
      result = result.filter((run) => run.status === statusFilter)
    }

    // Apply model filter
    if (modelFilter.length > 0) {
      result = result.filter((run) => modelFilter.includes(run.modelId))
    }

    // Apply sorting
    result.sort((a, b) => {
      let valueA, valueB

      switch (sortBy) {
        case "timestamp":
          valueA = new Date(a.timestamp).getTime()
          valueB = new Date(b.timestamp).getTime()
          break
        case "rmse":
          valueA = a.rmse || Number.MAX_VALUE
          valueB = b.rmse || Number.MAX_VALUE
          break
        case "duration":
          valueA = a.duration
          valueB = b.duration
          break
        default:
          valueA = new Date(a.timestamp).getTime()
          valueB = new Date(b.timestamp).getTime()
      }

      return sortOrder === "asc" ? valueA - valueB : valueB - valueA
    })

    setFilteredRuns(result)
  }, [runs, searchQuery, statusFilter, modelFilter, sortBy, sortOrder])

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "FINISHED":
        return <Badge className="bg-green-500">Finished</Badge>
      case "RUNNING":
        return <Badge className="bg-blue-500">Running</Badge>
      case "FAILED":
        return <Badge className="bg-red-500">Failed</Badge>
      default:
        return <Badge className="bg-gray-500">Unknown</Badge>
    }
  }

  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    return date.toLocaleString()
  }

  const formatDuration = (seconds: number) => {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    const secs = seconds % 60

    return `${hours > 0 ? `${hours}h ` : ""}${minutes}m ${secs}s`
  }

  const handleViewRun = (run: MLflowRun) => {
    setSelectedRun(run)
    setActiveTab("overview")
  }

  const handleDownloadArtifact = (runId: string, artifactName: string) => {
    // In a real implementation, this would download the artifact via API
    console.log(`Downloading artifact ${artifactName} for run ${runId}`)
    // Show success message
    alert(`Artifact ${artifactName} downloaded successfully`)
  }

  const handleRefreshRuns = () => {
    // In a real implementation, this would refresh the runs via API
    setLoading(true)
    setTimeout(() => {
      setLoading(false)
    }, 1000)
  }

  const clearFilters = () => {
    setSearchQuery("")
    setStatusFilter("all")
    setModelFilter([])
    setSortBy("timestamp")
    setSortOrder("desc")
  }

  const uniqueModels = Array.from(new Set(runs.map((run) => run.modelId)))

  return (
    <>
      <Card>
        <CardHeader className="pb-2">
          <div className="flex justify-between items-center">
            <div>
              <CardTitle className="text-lg font-medium">Recent MLflow Runs</CardTitle>
              <CardDescription>Latest model training runs across all models</CardDescription>
            </div>
            <Button variant="outline" size="sm" onClick={handleRefreshRuns}>
              <RefreshCw className="h-4 w-4 mr-1" />
              Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="mb-4 flex flex-col gap-2 sm:flex-row">
            <div className="relative flex-1">
              <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search runs..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-8"
              />
            </div>

            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger className="w-full sm:w-[180px]">
                <SelectValue placeholder="Filter by status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Statuses</SelectItem>
                <SelectItem value="FINISHED">Finished</SelectItem>
                <SelectItem value="RUNNING">Running</SelectItem>
                <SelectItem value="FAILED">Failed</SelectItem>
              </SelectContent>
            </Select>

            <Popover>
              <PopoverTrigger asChild>
                <Button variant="outline" className="w-full sm:w-[180px] justify-between">
                  <span>
                    {modelFilter.length > 0
                      ? `${modelFilter.length} model${modelFilter.length > 1 ? "s" : ""}`
                      : "Filter by model"}
                  </span>
                  <Filter className="h-4 w-4 ml-2" />
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-[200px] p-0" align="end">
                <div className="p-4 space-y-2">
                  {uniqueModels.map((model) => (
                    <div key={model} className="flex items-center space-x-2">
                      <Checkbox
                        id={`model-${model}`}
                        checked={modelFilter.includes(model)}
                        onCheckedChange={(checked) => {
                          if (checked) {
                            setModelFilter([...modelFilter, model])
                          } else {
                            setModelFilter(modelFilter.filter((m) => m !== model))
                          }
                        }}
                      />
                      <Label htmlFor={`model-${model}`}>{model}</Label>
                    </div>
                  ))}

                  <Button variant="ghost" size="sm" className="w-full mt-2" onClick={() => setModelFilter([])}>
                    Clear selection
                  </Button>
                </div>
              </PopoverContent>
            </Popover>

            <Select value={sortBy} onValueChange={setSortBy}>
              <SelectTrigger className="w-full sm:w-[180px]">
                <SelectValue placeholder="Sort by" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="timestamp">Date</SelectItem>
                <SelectItem value="rmse">RMSE</SelectItem>
                <SelectItem value="duration">Duration</SelectItem>
              </SelectContent>
            </Select>

            <Button
              variant="ghost"
              size="icon"
              onClick={() => setSortOrder(sortOrder === "asc" ? "desc" : "asc")}
              className="hidden sm:flex"
            >
              {sortOrder === "asc" ? (
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="24"
                  height="24"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="lucide lucide-arrow-up-narrow-wide"
                >
                  <path d="m3 8 4-4 4 4" />
                  <path d="M7 4v16" />
                  <path d="M11 12h4" />
                  <path d="M11 16h7" />
                  <path d="M11 20h10" />
                </svg>
              ) : (
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="24"
                  height="24"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="lucide lucide-arrow-down-narrow-wide"
                >
                  <path d="m3 16 4 4 4-4" />
                  <path d="M7 20V4" />
                  <path d="M11 4h4" />
                  <path d="M11 8h7" />
                  <path d="M11 12h10" />
                </svg>
              )}
            </Button>
          </div>

          {loading ? (
            <Skeleton className="h-[220px] w-full" />
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Run ID</TableHead>
                  <TableHead>Model</TableHead>
                  <TableHead>RMSE</TableHead>
                  <TableHead>Timestamp</TableHead>
                  <TableHead>Duration</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredRuns.length > 0 ? (
                  filteredRuns.map((run) => (
                    <TableRow key={run.runId}>
                      <TableCell className="font-mono text-xs">{run.runId.substring(0, 8)}...</TableCell>
                      <TableCell>{run.modelId}</TableCell>
                      <TableCell>{run.rmse ? run.rmse.toFixed(4) : "N/A"}</TableCell>
                      <TableCell className="text-sm">{formatDate(run.timestamp)}</TableCell>
                      <TableCell>{formatDuration(run.duration)}</TableCell>
                      <TableCell>{getStatusBadge(run.status)}</TableCell>
                      <TableCell className="text-right">
                        <Button variant="ghost" size="sm" className="h-6 px-2" onClick={() => handleViewRun(run)}>
                          <Eye className="h-3.5 w-3.5 mr-1" />
                          <span className="text-xs">View</span>
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))
                ) : (
                  <TableRow>
                    <TableCell colSpan={7} className="text-center py-8 text-muted-foreground">
                      No runs match your filters. Try adjusting your search criteria.
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          )}
        </CardContent>
        <CardFooter className="pt-0">
          <div className="w-full flex justify-between items-center">
            <div className="text-sm text-muted-foreground">
              Showing {filteredRuns.length} of {runs.length} runs
              {(searchQuery || statusFilter !== "all" || modelFilter.length > 0) && (
                <Button variant="ghost" size="sm" className="ml-2" onClick={clearFilters}>
                  Clear filters
                </Button>
              )}
            </div>
            <Button variant="link" className="ml-auto">
              View All Runs
            </Button>
          </div>
        </CardFooter>
      </Card>

      {/* Run Details Dialog */}
      {selectedRun && (
        <Dialog open={!!selectedRun} onOpenChange={(open) => !open && setSelectedRun(null)}>
          <DialogContent className="max-w-4xl">
            <DialogHeader>
              <DialogTitle className="flex items-center">
                Run Details: {selectedRun.runId}
                {getStatusBadge(selectedRun.status)}
              </DialogTitle>
              <DialogDescription>
                Model: {selectedRun.modelId} | Started: {formatDate(selectedRun.timestamp)}
              </DialogDescription>
            </DialogHeader>

            <Tabs defaultValue="overview" value={activeTab} onValueChange={setActiveTab}>
              <TabsList className="grid w-full grid-cols-4">
                <TabsTrigger value="overview">Overview</TabsTrigger>
                <TabsTrigger value="parameters">Parameters</TabsTrigger>
                <TabsTrigger value="artifacts">Artifacts</TabsTrigger>
                <TabsTrigger value="logs">Logs</TabsTrigger>
              </TabsList>

              <TabsContent value="overview" className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <h3 className="text-sm font-medium mb-2">Run Information</h3>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-xs text-muted-foreground">Run ID</span>
                        <span className="text-xs font-medium font-mono">{selectedRun.runId}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-xs text-muted-foreground">Model</span>
                        <span className="text-xs font-medium">{selectedRun.modelId}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-xs text-muted-foreground">Status</span>
                        <span className="text-xs font-medium">{selectedRun.status}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-xs text-muted-foreground">Start Time</span>
                        <span className="text-xs font-medium">{formatDate(selectedRun.timestamp)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-xs text-muted-foreground">Duration</span>
                        <span className="text-xs font-medium">{formatDuration(selectedRun.duration)}</span>
                      </div>
                    </div>
                  </div>

                  <div>
                    <h3 className="text-sm font-medium mb-2">Metrics</h3>
                    <div className="space-y-2">
                      {selectedRun.metrics &&
                        Object.entries(selectedRun.metrics).map(([key, value]) => (
                          <div key={key} className="flex justify-between">
                            <span className="text-xs text-muted-foreground">{key.toUpperCase()}</span>
                            <span className="text-xs font-medium">
                              {typeof value === "number" ? value.toFixed(4) : value}
                            </span>
                          </div>
                        ))}
                      {(!selectedRun.metrics || Object.keys(selectedRun.metrics).length === 0) && (
                        <p className="text-xs text-muted-foreground">No metrics available</p>
                      )}
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="text-sm font-medium mb-2">Tags</h3>
                  <div className="flex flex-wrap gap-2">
                    {selectedRun.tags &&
                      Object.entries(selectedRun.tags).map(([key, value]) => (
                        <Badge key={key} variant="outline" className="text-xs">
                          {key}: {value}
                        </Badge>
                      ))}
                    {(!selectedRun.tags || Object.keys(selectedRun.tags).length === 0) && (
                      <p className="text-xs text-muted-foreground">No tags available</p>
                    )}
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="parameters">
                <div className="max-h-[400px] overflow-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Parameter</TableHead>
                        <TableHead>Value</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {selectedRun.parameters &&
                        Object.entries(selectedRun.parameters).map(([key, value]) => (
                          <TableRow key={key}>
                            <TableCell className="font-medium">{key}</TableCell>
                            <TableCell>{JSON.stringify(value)}</TableCell>
                          </TableRow>
                        ))}
                      {(!selectedRun.parameters || Object.keys(selectedRun.parameters).length === 0) && (
                        <TableRow>
                          <TableCell colSpan={2} className="text-center text-muted-foreground">
                            No parameters available
                          </TableCell>
                        </TableRow>
                      )}
                    </TableBody>
                  </Table>
                </div>
              </TabsContent>

              <TabsContent value="artifacts">
                <div className="max-h-[400px] overflow-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Name</TableHead>
                        <TableHead>Type</TableHead>
                        <TableHead>Size</TableHead>
                        <TableHead>Last Modified</TableHead>
                        <TableHead className="text-right">Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {selectedRun.artifacts &&
                        selectedRun.artifacts.map((artifact) => (
                          <TableRow key={artifact.name}>
                            <TableCell className="font-medium">{artifact.name}</TableCell>
                            <TableCell>{artifact.type}</TableCell>
                            <TableCell>{artifact.size}</TableCell>
                            <TableCell>{new Date(artifact.lastModified).toLocaleString()}</TableCell>
                            <TableCell className="text-right">
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => handleDownloadArtifact(selectedRun.runId, artifact.name)}
                              >
                                <Download className="h-4 w-4 mr-1" />
                                <span className="text-xs">Download</span>
                              </Button>
                            </TableCell>
                          </TableRow>
                        ))}
                      {(!selectedRun.artifacts || selectedRun.artifacts.length === 0) && (
                        <TableRow>
                          <TableCell colSpan={5} className="text-center text-muted-foreground">
                            No artifacts available
                          </TableCell>
                        </TableRow>
                      )}
                    </TableBody>
                  </Table>
                </div>
              </TabsContent>

              <TabsContent value="logs">
                <div className="max-h-[400px] overflow-auto">
                  <pre className="bg-muted p-4 rounded-md text-xs font-mono whitespace-pre-wrap">
                    {selectedRun.logs || "No logs available"}
                  </pre>
                </div>
              </TabsContent>
            </Tabs>

            <DialogFooter className="flex justify-between items-center">
              <div className="flex space-x-2">
                <Button variant="outline" size="sm">
                  <LineChart className="h-4 w-4 mr-1" />
                  Metrics
                </Button>
                <Button variant="outline" size="sm">
                  <BarChart className="h-4 w-4 mr-1" />
                  Plots
                </Button>
              </div>
              <div className="flex space-x-2">
                <Button variant="outline" size="sm">
                  <Download className="h-4 w-4 mr-1" />
                  Export
                </Button>
                <Button variant="default" size="sm">
                  <Play className="h-4 w-4 mr-1" />
                  Rerun
                </Button>
              </div>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      )}
    </>
  )
}
