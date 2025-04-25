"use client"

import { useEffect, useState } from "react"
import { CheckCircle, Clock, AlertTriangle, XCircle, Play, Pause, RefreshCw, Eye, Search } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog"
import { Table, TableHeader, TableRow, TableCell, TableBody, TableHeadCell } from "@/components/ui/table"

interface PipelineRun {
  dagId: string
  runId: string
  status: "success" | "running" | "failed" | "warning"
  startDate: string
  endDate: string | null
  duration: number
  sla: number
  nextRun: string
  tasks?: PipelineTask[]
  logs?: string
}

interface PipelineTask {
  taskId: string
  status: "success" | "running" | "failed" | "upstream_failed" | "skipped"
  startDate: string | null
  endDate: string | null
  duration: number | null
  tries: number
  maxTries: number
}

export function PipelineHealth() {
  const [loading, setLoading] = useState(true)
  const [pipelineData, setPipelineData] = useState<PipelineRun[]>([])
  const [filteredData, setFilteredData] = useState<PipelineRun[]>([])
  const [selectedPipeline, setSelectedPipeline] = useState<PipelineRun | null>(null)
  const [activeTab, setActiveTab] = useState("tasks")

  // Filter states
  const [searchQuery, setSearchQuery] = useState("")
  const [statusFilter, setStatusFilter] = useState<string>("all")
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc")

  useEffect(() => {
    // Simulate API call to Airflow
    const fetchPipelineHealth = async () => {
      // In a real implementation, this would be an API call to our internal API that fetches from Airflow
      // No external links needed as everything is integrated

      // Simulated data
      setTimeout(() => {
        const mockData: PipelineRun[] = [
          {
            dagId: "loss_history_etl",
            runId: "scheduled__2023-04-15T01:00:00+00:00",
            status: "success",
            startDate: "2023-04-15T01:00:00+00:00",
            endDate: "2023-04-15T01:45:23+00:00",
            duration: 2723,
            sla: 3600,
            nextRun: "2023-04-16T01:00:00+00:00",
            tasks: [
              {
                taskId: "extract_data",
                status: "success",
                startDate: "2023-04-15T01:00:05+00:00",
                endDate: "2023-04-15T01:15:23+00:00",
                duration: 918,
                tries: 1,
                maxTries: 3,
              },
              {
                taskId: "transform_data",
                status: "success",
                startDate: "2023-04-15T01:15:30+00:00",
                endDate: "2023-04-15T01:35:45+00:00",
                duration: 1215,
                tries: 1,
                maxTries: 3,
              },
              {
                taskId: "load_data",
                status: "success",
                startDate: "2023-04-15T01:36:00+00:00",
                endDate: "2023-04-15T01:45:23+00:00",
                duration: 563,
                tries: 1,
                maxTries: 3,
              }
            ],
            logs: "INFO - Starting DAG run loss_history_etl\nINFO - Task extract_data started\nINFO - Extracting data from source systems\nINFO - Extracted 24,532 records\nINFO - Task extract_data completed successfully\nINFO - Task transform_data started\nINFO - Applying transformations to data\nINFO - Cleaned 152 records with missing values\nINFO - Normalized 24,532 records\nINFO - Task transform_data completed successfully\nINFO - Task load_data started\nINFO - Loading data to data warehouse\nINFO - Loaded 24,532 records successfully\nINFO - Task load_data completed successfully\nINFO - DAG run completed successfully",
          },
          {
            dagId: "model_training_pipeline",
            runId: "scheduled__2023-04-15T06:00:00+00:00",
            status: "warning",
            startDate: "2023-04-15T06:00:00+00:00",
            endDate: "2023-04-15T07:15:42+00:00",
            duration: 4542,
            sla: 4500,
            nextRun: "2023-04-16T06:00:00+00:00",
            tasks: [
              {
                taskId: "prepare_features",
                status: "success",
                startDate: "2023-04-15T06:00:05+00:00",
                endDate: "2023-04-15T06:25:30+00:00",
                duration: 1525,
                tries: 1,
                maxTries: 3,
              },
              {
                taskId: "train_model",
                status: "success",
                startDate: "2023-04-15T06:25:45+00:00",
                endDate: "2023-04-15T07:10:15+00:00",
                duration: 2670,
                tries: 2,
                maxTries: 3,
              },
              {
                taskId: "evaluate_model",
                status: "success",
                startDate: "2023-04-15T07:10:30+00:00",
                endDate: "2023-04-15T07:15:42+00:00",
                duration: 312,
                tries: 1,
                maxTries: 3,
              },
            ],
            logs: "INFO - Starting DAG run model_training_pipeline\nINFO - Task prepare_features started\nINFO - Preparing features for model training\nINFO - Generated 42 features for training\nINFO - Task prepare_features completed successfully\nINFO - Task train_model started\nWARNING - Memory usage high during training\nERROR - Out of memory error, retrying with reduced batch size\nINFO - Retrying task train_model\nINFO - Model training completed with 500 iterations\nINFO - Task train_model completed successfully\nINFO - Task evaluate_model started\nINFO - Evaluating model performance\nINFO - Model achieved 94.2% accuracy\nINFO - Task evaluate_model completed successfully\nWARNING - DAG run exceeded SLA by 42 seconds",
          },
          {
            dagId: "feature_engineering",
            runId: "scheduled__2023-04-15T04:00:00+00:00",
            status: "success",
            startDate: "2023-04-15T04:00:00+00:00",
            endDate: "2023-04-15T04:32:18+00:00",
            duration: 1938,
            sla: 2400,
            nextRun: "2023-04-16T04:00:00+00:00",
            tasks: [
              {
                taskId: "extract_raw_features",
                status: "success",
                startDate: "2023-04-15T04:00:05+00:00",
                endDate: "2023-04-15T04:10:30+00:00",
                duration: 625,
                tries: 1,
                maxTries: 3,
              },
              {
                taskId: "create_derived_features",
                status: "success",
                startDate: "2023-04-15T04:10:45+00:00",
                endDate: "2023-04-15T04:25:15+00:00",
                duration: 870,
                tries: 1,
                maxTries: 3,
              },
              {
                taskId: "save_feature_store",
                status: "success",
                startDate: "2023-04-15T04:25:30+00:00",
                endDate: "2023-04-15T04:32:18+00:00",
                duration: 408,
                tries: 1,
                maxTries: 3,
              },
            ],
            logs: "INFO - Starting DAG run feature_engineering\nINFO - Task extract_raw_features started\nINFO - Extracting raw features from data sources\nINFO - Extracted 32 raw features\nINFO - Task extract_raw_features completed successfully\nINFO - Task create_derived_features started\nINFO - Creating derived features\nINFO - Created 10 new derived features\nINFO - Task create_derived_features completed successfully\nINFO - Task save_feature_store started\nINFO - Saving features to feature store\nINFO - Saved 42 features to feature store\nINFO - Task save_feature_store completed successfully\nINFO - DAG run completed successfully",
          },
          {
            dagId: "data_quality_checks",
            runId: "scheduled__2023-04-15T00:30:00+00:00",
            status: "running",
            startDate: "2023-04-15T00:30:00+00:00",
            endDate: null,
            duration: 1200, // Current duration
            sla: 1800,
            nextRun: "2023-04-16T00:30:00+00:00",
            tasks: [
              {
                taskId: "check_completeness",
                status: "success",
                startDate: "2023-04-15T00:30:05+00:00",
                endDate: "2023-04-15T00:45:30+00:00",
                duration: 925,
                tries: 1,
                maxTries: 3,
              },
              {
                taskId: "check_consistency",
                status: "running",
                startDate: "2023-04-15T00:45:45+00:00",
                endDate: null,
                duration: null,
                tries: 1,
                maxTries: 3,
              },
              {
                taskId: "generate_quality_report",
                status: "skipped",
                startDate: null,
                endDate: null,
                duration: null,
                tries: 0,
                maxTries: 3,
              },
            ],
            logs: "INFO - Starting DAG run data_quality_checks\nINFO - Task check_completeness started\nINFO - Checking data completeness\nINFO - Found 99.7% completeness across all fields\nINFO - Task check_completeness completed successfully\nINFO - Task check_consistency started\nINFO - Checking data consistency\nINFO - Currently validating cross-field consistency rules\nINFO - 45% complete...",
          },
          {
            dagId: "model_evaluation",
            runId: "scheduled__2023-04-14T12:00:00+00:00",
            status: "failed",
            startDate: "2023-04-14T12:00:00+00:00",
            endDate: "2023-04-14T12:15:30+00:00",
            duration: 930,
            sla: 1800,
            nextRun: "2023-04-15T12:00:00+00:00",
            tasks: [
              {
                taskId: "load_model",
                status: "success",
                startDate: "2023-04-14T12:00:05+00:00",
                endDate: "2023-04-14T12:05:30+00:00",
                duration: 325,
                tries: 1,
                maxTries: 3,
              },
              {
                taskId: "load_test_data",
                status: "success",
                startDate: "2023-04-14T12:05:45+00:00",
                endDate: "2023-04-14T12:10:15+00:00",
                duration: 270,
                tries: 1,
                maxTries: 3,
              },
              {
                taskId: "evaluate_model",
                status: "failed",
                startDate: "2023-04-14T12:10:30+00:00",
                endDate: "2023-04-14T12:15:30+00:00",
                duration: 300,
                tries: 3,
                maxTries: 3,
              },
            ],
            logs: "INFO - Starting DAG run model_evaluation\nINFO - Task load_model started\nINFO - Loading model from model registry\nINFO - Model loaded successfully\nINFO - Task load_model completed successfully\nINFO - Task load_test_data started\nINFO - Loading test data from data warehouse\nINFO - Test data loaded successfully\nINFO - Task load_test_data completed successfully\nINFO - Task evaluate_model started\nERROR - Invalid model format\nERROR - Retrying task evaluate_model\nERROR - Invalid model format\nERROR - Retrying task evaluate_model\nERROR - Invalid model format\nERROR - Task evaluate_model failed after 3 retries\nERROR - DAG run failed",
          },
        ]

        setPipelineData(mockData)
        setFilteredData(mockData)
        setLoading(false)
      }, 1000)
    }

    fetchPipelineHealth()
  }, [])

  // Apply filters whenever filter states change
  useEffect(() => {
    let result = [...pipelineData]

    // Apply search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      result = result.filter(
        (pipeline) => pipeline.dagId.toLowerCase().includes(query) || pipeline.runId.toLowerCase().includes(query),
      )
    }

    // Apply status filter
    if (statusFilter !== "all") {
      result = result.filter((pipeline) => pipeline.status === statusFilter)
    }

    // Apply sorting
    result.sort((a, b) => {
      const dateA = new Date(a.startDate).getTime()
      const dateB = new Date(b.startDate).getTime()
      return sortOrder === "asc" ? dateA - dateB : dateB - dateA
    })

    setFilteredData(result)
  }, [pipelineData, searchQuery, statusFilter, sortOrder])

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "success":
        return <CheckCircle className="h-5 w-5 text-green-500" />
      case "running":
        return <Clock className="h-5 w-5 text-blue-500 animate-pulse" />
      case "warning":
        return <AlertTriangle className="h-5 w-5 text-yellow-500" />
      case "failed":
        return <XCircle className="h-5 w-5 text-red-500" />
      case "upstream_failed":
        return <XCircle className="h-5 w-5 text-orange-500" />
      case "skipped":
        return <AlertTriangle className="h-5 w-5 text-gray-500" />
      default:
        return null
    }
  }

  const formatDuration = (seconds: number | null) => {
    if (seconds === null) return "N/A"

    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    const secs = seconds % 60

    return `${hours > 0 ? `${hours}h ` : ""}${minutes}m ${secs}s`
  }

  const formatDate = (dateString: string | null) => {
    if (!dateString) return "N/A"
    const date = new Date(dateString)
    return date.toLocaleString()
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case "success":
        return "#10b981" // green
      case "running":
        return "#3b82f6" // blue
      case "warning":
        return "#f59e0b" // yellow
      case "failed":
        return "#ef4444" // red
      default:
        return "#6b7280" // gray
    }
  }

  const handleViewDetails = (pipeline: PipelineRun) => {
    setSelectedPipeline(pipeline)
  }

  const handleRunPipeline = (dagId: string) => {
    // In a real implementation, this would trigger the pipeline via API
    console.log(`Triggering pipeline: ${dagId}`)
    // Show success message
    alert(`Pipeline ${dagId} triggered successfully`)
  }

  const handlePausePipeline = (dagId: string) => {
    // In a real implementation, this would pause the pipeline via API
    console.log(`Pausing pipeline: ${dagId}`)
    // Show success message
    alert(`Pipeline ${dagId} paused successfully`)
  }

  const handleRefreshPipeline = (dagId: string) => {
    // In a real implementation, this would refresh the pipeline status via API
    console.log(`Refreshing pipeline: ${dagId}`)
    setLoading(true)
    setTimeout(() => {
      setLoading(false)
    }, 1000)
  }

  return (
    <>
      <Card>
        <CardHeader className="pb-2">
          <div className="flex justify-between items-center">
            <div>
              <CardTitle className="text-lg font-medium">Pipeline Health</CardTitle>
              <CardDescription>Status of recent DAG runs from Airflow</CardDescription>
            </div>
            <Button variant="outline" size="sm" onClick={() => handleRefreshPipeline("")}>
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
                placeholder="Search pipelines..."
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
                <SelectItem value="success">Success</SelectItem>
                <SelectItem value="running">Running</SelectItem>
                <SelectItem value="warning">Warning</SelectItem>
                <SelectItem value="failed">Failed</SelectItem>
              </SelectContent>
            </Select>

            <Select value={sortOrder} onValueChange={(value) => setSortOrder(value as "asc" | "desc")}>
              <SelectTrigger className="w-full sm:w-[180px]">
                <SelectValue placeholder="Sort by" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="desc">Newest First</SelectItem>
                <SelectItem value="asc">Oldest First</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {loading ? (
            <div className="space-y-3">
              {[1, 2, 3].map((i) => (
                <div key={i} className="flex items-center space-x-4">
                  <Skeleton className="h-10 w-10 rounded-full" />
                  <div className="space-y-2">
                    <Skeleton className="h-4 w-[250px]" />
                    <Skeleton className="h-4 w-[200px]" />
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="space-y-3">
              {filteredData.length > 0 ? (
                filteredData.map((pipeline) => (
                  <Card
                    key={pipeline.dagId}
                    className={`p-3 overflow-hidden border-l-4 ${
                      pipeline.status === "running" ? "border-blue-500" :
                      pipeline.status === "success" ? "border-green-500" :
                      pipeline.status === "warning" ? "border-yellow-500" :
                      pipeline.status === "failed" ? "border-red-500" : "border-gray-500"
                    } ${pipeline.status === "running" ? "animate-pulse" : ""}`}
                  >
                    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                      <div className="flex items-center space-x-3">
                        {getStatusIcon(pipeline.status)}
                        <div>
                          <h4 className="font-medium truncate">{pipeline.dagId}</h4>
                          <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
                            <span className="truncate">
                              Last run: {pipeline.endDate ? formatDate(pipeline.endDate) : "In progress"}
                            </span>
                            <Badge variant={pipeline.duration > pipeline.sla ? "destructive" : "outline"}>
                              {formatDuration(pipeline.duration)} / {formatDuration(pipeline.sla)}
                            </Badge>
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center justify-between sm:justify-end gap-2">
                        <div className="text-sm">
                          <div className="text-right">
                            <span className="text-muted-foreground">Next run:</span>
                          </div>
                          <div className="font-medium">{formatDate(pipeline.nextRun)}</div>
                        </div>
                        <div className="flex space-x-1 flex-shrink-0">
                          <Button variant="ghost" size="icon" onClick={() => handleViewDetails(pipeline)}>
                            <Eye className="h-4 w-4" />
                          </Button>
                          <Button variant="ghost" size="icon" onClick={() => handleRunPipeline(pipeline.dagId)}>
                            <Play className="h-4 w-4" />
                          </Button>
                          <Button variant="ghost" size="icon" onClick={() => handlePausePipeline(pipeline.dagId)}>
                            <Pause className="h-4 w-4" />
                          </Button>
                          <Button variant="ghost" size="icon" onClick={() => handleRefreshPipeline(pipeline.dagId)}>
                            <RefreshCw className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    </div>
                  </Card>
                ))
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  No pipelines match your filters. Try adjusting your search criteria.
                </div>
              )}
            </div>
          )}
        </CardContent>

        <CardFooter className="pt-0">
          <div className="w-full flex justify-between items-center">
            <div className="text-sm text-muted-foreground">
              Showing {filteredData.length} of {pipelineData.length} pipelines
            </div>
            <Button variant="link">View All Pipelines</Button>
          </div>
        </CardFooter>
      </Card>

      {/* Pipeline Details Dialog */}
      {selectedPipeline && (
        <Dialog open={!!selectedPipeline} onOpenChange={(open) => !open && setSelectedPipeline(null)}>
          <DialogContent className="max-w-4xl">
            <DialogHeader>
              <DialogTitle className="flex items-center">
                {getStatusIcon(selectedPipeline.status)}
                <span className="ml-2">{selectedPipeline.dagId}</span>
                <Badge
                  className="ml-2"
                  variant={
                    selectedPipeline.status === "success"
                      ? "default"
                      : selectedPipeline.status === "warning"
                        ? "outline"
                        : "destructive"
                  }
                >
                  {selectedPipeline.status}
                </Badge>
              </DialogTitle>
              <DialogDescription>Run ID: {selectedPipeline.runId}</DialogDescription>
            </DialogHeader>

            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <p className="text-sm text-muted-foreground">Start Time</p>
                <p className="font-medium">{formatDate(selectedPipeline.startDate)}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">End Time</p>
                <p className="font-medium">
                  {selectedPipeline.endDate ? formatDate(selectedPipeline.endDate) : "In progress"}
                </p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Duration</p>
                <p className="font-medium">{formatDuration(selectedPipeline.duration)}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">SLA</p>
                <p className="font-medium">{formatDuration(selectedPipeline.sla)}</p>
              </div>
            </div>

            <Tabs defaultValue="tasks" value={activeTab} onValueChange={setActiveTab}>
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="tasks">Tasks</TabsTrigger>
                <TabsTrigger value="logs">Logs</TabsTrigger>
              </TabsList>
              <TabsContent value="tasks" className="max-h-[400px] overflow-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHeadCell>Task ID</TableHeadCell>
                      <TableHeadCell>Status</TableHeadCell>
                      <TableHeadCell>Start Time</TableHeadCell>
                      <TableHeadCell>End Time</TableHeadCell>
                      <TableHeadCell>Duration</TableHeadCell>
                      <TableHeadCell>Tries</TableHeadCell>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {selectedPipeline.tasks?.map((task) => (
                      <TableRow key={task.taskId}>
                        <TableCell>{task.taskId}</TableCell>
                        <TableCell>
                          <div className="flex items-center">
                            {getStatusIcon(task.status)}
                            <span className="ml-2 capitalize">{task.status}</span>
                          </div>
                        </TableCell>
                        <TableCell>{formatDate(task.startDate)}</TableCell>
                        <TableCell>{formatDate(task.endDate)}</TableCell>
                        <TableCell>{formatDuration(task.duration)}</TableCell>
                        <TableCell>
                          {task.tries} / {task.maxTries}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TabsContent>
              <TabsContent value="logs" className="max-h-[400px] overflow-auto">
                <pre className="bg-muted p-4 rounded-md text-xs font-mono whitespace-pre-wrap">
                  {selectedPipeline.logs}
                </pre>
              </TabsContent>
            </Tabs>

            <DialogFooter className="flex justify-between items-center">
              <div>
                <p className="text-sm text-muted-foreground">Next Run</p>
                <p className="font-medium">{formatDate(selectedPipeline.nextRun)}</p>
              </div>
              <div className="flex space-x-2">
                <Button variant="outline" onClick={() => handleRunPipeline(selectedPipeline.dagId)}>
                  <Play className="h-4 w-4 mr-2" />
                  Run Now
                </Button>
                <Button variant="default">
                  <Eye className="h-4 w-4 mr-2" />
                  View in Graph
                </Button>
              </div>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      )}
    </>
  )
}
