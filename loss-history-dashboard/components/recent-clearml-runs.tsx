"use client"

import { useEffect, useState } from "react"
import { Download, Play, RefreshCw, BarChart, LineChart, Eye, Search, Filter, ExternalLink } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableHeader, TableRow, TableCell, TableBody, TableHeadCell } from "@/components/ui/table"
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

interface ClearMLTask {
  id: string
  name: string
  type: string
  status: string
  created: string
  started: string
  completed?: string
  project_id: string
  project_name: string
  user: string
  metrics?: Record<string, number>
  hyperparams?: Record<string, any>
  tags?: string[]
  artifacts?: Record<string, any>
  web_url?: string
}

export function RecentClearMLRuns() {
  const [loading, setLoading] = useState(true)
  const [tasks, setTasks] = useState<ClearMLTask[]>([])
  const [filteredTasks, setFilteredTasks] = useState<ClearMLTask[]>([])
  const [selectedTask, setSelectedTask] = useState<ClearMLTask | null>(null)
  const [activeTab, setActiveTab] = useState("overview")
  const [projectId, setProjectId] = useState("")
  const [projects, setProjects] = useState<{id: string, name: string}[]>([])

  // Filter states
  const [searchQuery, setSearchQuery] = useState("")
  const [statusFilter, setStatusFilter] = useState<string>("all")
  const [typeFilter, setTypeFilter] = useState<string[]>([])
  const [sortBy, setSortBy] = useState<string>("last_update")
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc")

  useEffect(() => {
    // Fetch projects first to populate the dropdown
    const fetchProjects = async () => {
      try {
        const response = await fetch(`/api/clearml/tasks?limit=1`);
        if (!response.ok) throw new Error('Failed to fetch initial project');
        
        const data = await response.json();
        if (data.project_id) {
          setProjectId(data.project_id);
          setProjects([{ id: data.project_id, name: data.project_name }]);
          fetchTasksForProject(data.project_id);
        }
      } catch (error) {
        console.error("Error fetching projects:", error);
        setLoading(false);
      }
    };

    fetchProjects();
  }, []);

  const fetchTasksForProject = async (pid: string) => {
    setLoading(true);
    try {
      const response = await fetch(`/api/clearml/tasks?project_id=${pid}&limit=100`);
      if (!response.ok) throw new Error('Failed to fetch tasks');
      
      const data = await response.json();
      setTasks(data.tasks || []);
      setFilteredTasks(data.tasks || []);
    } catch (error) {
      console.error("Error fetching tasks:", error);
    } finally {
      setLoading(false);
    }
  };

  // Apply filters whenever filter states change
  useEffect(() => {
    let result = [...tasks]

    // Apply search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      result = result.filter(
        (task) =>
          task.id.toLowerCase().includes(query) ||
          task.name.toLowerCase().includes(query) ||
          task.user.toLowerCase().includes(query)
      )
    }

    // Apply status filter
    if (statusFilter !== "all") {
      result = result.filter((task) => task.status === statusFilter)
    }

    // Apply type filter
    if (typeFilter.length > 0) {
      result = result.filter((task) => typeFilter.includes(task.type))
    }

    // Apply sorting
    result.sort((a, b) => {
      let valueA, valueB;

      switch (sortBy) {
        case "last_update":
          valueA = new Date(a.created).getTime();
          valueB = new Date(b.created).getTime();
          break;
        case "name":
          valueA = a.name;
          valueB = b.name;
          break;
        case "type":
          valueA = a.type;
          valueB = b.type;
          break;
        case "user":
          valueA = a.user;
          valueB = b.user;
          break;
        default:
          valueA = new Date(a.created).getTime();
          valueB = new Date(b.created).getTime();
      }

      if (typeof valueA === 'string' && typeof valueB === 'string') {
        return sortOrder === "asc" 
          ? valueA.localeCompare(valueB) 
          : valueB.localeCompare(valueA);
      }

      return sortOrder === "asc" ? valueA - valueB : valueB - valueA;
    });

    setFilteredTasks(result);
  }, [tasks, searchQuery, statusFilter, typeFilter, sortBy, sortOrder]);

  const getStatusBadge = (status: string) => {
    switch (status.toLowerCase()) {
      case "completed":
        return <Badge className="bg-green-500">Completed</Badge>
      case "in_progress":
        return <Badge className="bg-blue-500">Running</Badge>
      case "failed":
        return <Badge className="bg-red-500">Failed</Badge>
      case "created":
        return <Badge className="bg-gray-500">Created</Badge>
      case "queued":
        return <Badge className="bg-amber-500">Queued</Badge>
      default:
        return <Badge className="bg-gray-500">{status}</Badge>
    }
  };

  const formatDate = (dateString: string) => {
    if (!dateString) return "N/A";
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  const formatDuration = (started?: string, completed?: string) => {
    if (!started) return "N/A";
    
    const startTime = new Date(started).getTime();
    const endTime = completed ? new Date(completed).getTime() : Date.now();
    
    const durationMs = endTime - startTime;
    const seconds = Math.floor(durationMs / 1000);
    
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;

    return `${hours > 0 ? `${hours}h ` : ""}${minutes}m ${secs}s`;
  };

  const handleViewTask = async (task: ClearMLTask) => {
    // Fetch full task details including metrics
    try {
      const response = await fetch(`/api/clearml/tasks/${task.id}`);
      if (!response.ok) throw new Error('Failed to fetch task details');
      
      const taskDetails = await response.json();
      setSelectedTask({...task, ...taskDetails});
      setActiveTab("overview");
    } catch (error) {
      console.error(`Error fetching task details for ${task.id}:`, error);
      setSelectedTask(task);
    }
  };

  const handleRefreshTasks = () => {
    if (projectId) {
      fetchTasksForProject(projectId);
    }
  };

  const clearFilters = () => {
    setSearchQuery("");
    setStatusFilter("all");
    setTypeFilter([]);
    setSortBy("last_update");
    setSortOrder("desc");
  };

  // Extract unique task types for filtering
  const uniqueTypes = Array.from(new Set(tasks.map(task => task.type)));

  return (
    <>
      <Card>
        <CardHeader className="pb-2">
          <div className="flex justify-between items-center">
            <div>
              <CardTitle className="text-lg font-medium">Recent ClearML Tasks</CardTitle>
              <CardDescription>Latest model training and pipeline tasks</CardDescription>
            </div>
            <Button variant="outline" size="sm" onClick={handleRefreshTasks}>
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
                placeholder="Search tasks..."
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
                <SelectItem value="completed">Completed</SelectItem>
                <SelectItem value="in_progress">Running</SelectItem>
                <SelectItem value="created">Created</SelectItem>
                <SelectItem value="queued">Queued</SelectItem>
                <SelectItem value="failed">Failed</SelectItem>
              </SelectContent>
            </Select>

            <Popover>
              <PopoverTrigger asChild>
                <Button variant="outline" className="w-full sm:w-[180px] justify-between">
                  <span>
                    {typeFilter.length > 0
                      ? `${typeFilter.length} type${typeFilter.length > 1 ? "s" : ""}`
                      : "Filter by type"}
                  </span>
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-[200px] p-0" align="end">
                <div className="p-4 space-y-2">
                  {uniqueTypes.map((type) => (
                    <div key={type} className="flex items-center space-x-2">
                      <Checkbox
                        id={`type-${type}`}
                        checked={typeFilter.includes(type)}
                        onCheckedChange={(checked) => {
                          if (checked) {
                            setTypeFilter([...typeFilter, type]);
                          } else {
                            setTypeFilter(typeFilter.filter((t) => t !== type));
                          }
                        }}
                      />
                      <Label htmlFor={`type-${type}`}>{type}</Label>
                    </div>
                  ))}

                  <Button variant="ghost" size="sm" className="w-full mt-2" onClick={() => setTypeFilter([])}>
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
                <SelectItem value="last_update">Date</SelectItem>
                <SelectItem value="name">Name</SelectItem>
                <SelectItem value="type">Type</SelectItem>
                <SelectItem value="user">User</SelectItem>
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
                  <TableHeadCell>Task ID</TableHeadCell>
                  <TableHeadCell>Name</TableHeadCell>
                  <TableHeadCell>Type</TableHeadCell>
                  <TableHeadCell>Started</TableHeadCell>
                  <TableHeadCell>Duration</TableHeadCell>
                  <TableHeadCell>Status</TableHeadCell>
                  <TableHeadCell className="text-right">Actions</TableHeadCell>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredTasks.length > 0 ? (
                {filteredTasks.length > 0 ? (
                  filteredTasks.map((task) => (
                    <TableRow key={task.id}>
                      <TableCell className="font-mono text-xs">{task.id.substring(0, 8)}...</TableCell>
                      <TableCell>{task.name}</TableCell>
                      <TableCell>{task.type}</TableCell>
                      <TableCell className="text-sm">{formatDate(task.started || task.created)}</TableCell>
                      <TableCell>{formatDuration(task.started, task.completed)}</TableCell>
                      <TableCell>{getStatusBadge(task.status)}</TableCell>
                      <TableCell className="text-right">
                        <Button variant="ghost" size="sm" className="h-6 px-2" onClick={() => handleViewTask(task)}>
                          <Eye className="h-3.5 w-3.5 mr-1" />
                          <span className="text-xs">View</span>
                        </Button>
                        {task.web_url && (
                          <Button 
                            variant="ghost" 
                            size="sm" 
                            className="h-6 px-2 ml-1"
                            onClick={() => window.open(task.web_url, '_blank')}
                          >
                            <ExternalLink className="h-3.5 w-3.5" />
                          </Button>
                        )}
                      </TableCell>
                    </TableRow>
                  ))
                ) : (
                  <TableRow>
                    <TableCell colSpan={7} className="text-center py-8 text-muted-foreground">
                      No tasks match your filters. Try adjusting your search criteria.
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
              Showing {filteredTasks.length} of {tasks.length} tasks
              {(searchQuery || statusFilter !== "all" || typeFilter.length > 0) && (
                <Button variant="ghost" size="sm" className="ml-2" onClick={clearFilters}>
                  Clear filters
                </Button>
              )}
            </div>
            <Button variant="link" className="ml-auto" onClick={() => window.open(`${selectedTask?.web_url?.split('/task/')[0]}`, '_blank')}>
              View All in ClearML
            </Button>
          </div>
        </CardFooter>
      </Card>

      {/* Task Details Dialog */}
      {selectedTask && (
        <Dialog open={!!selectedTask} onOpenChange={(open) => !open && setSelectedTask(null)}>
          <DialogContent className="max-w-4xl">
            <DialogHeader>
              <DialogTitle className="flex items-center">
                {selectedTask.name}
                {getStatusBadge(selectedTask.status)}
              </DialogTitle>
              <DialogDescription>
                ID: {selectedTask.id} | Started: {formatDate(selectedTask.started || selectedTask.created)}
              </DialogDescription>
            </DialogHeader>

            <Tabs defaultValue="overview" value={activeTab} onValueChange={setActiveTab}>
              <TabsList className="grid w-full grid-cols-4">
                <TabsTrigger value="overview">Overview</TabsTrigger>
                <TabsTrigger value="metrics">Metrics</TabsTrigger>
                <TabsTrigger value="params">Parameters</TabsTrigger>
                <TabsTrigger value="artifacts">Artifacts</TabsTrigger>
              </TabsList>

              <TabsContent value="overview" className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <h3 className="text-sm font-medium mb-2">Task Information</h3>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-xs text-muted-foreground">Task ID</span>
                        <span className="text-xs font-medium font-mono">{selectedTask.id}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-xs text-muted-foreground">Type</span>
                        <span className="text-xs font-medium">{selectedTask.type}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-xs text-muted-foreground">Status</span>
                        <span className="text-xs font-medium">{selectedTask.status}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-xs text-muted-foreground">Start Time</span>
                        <span className="text-xs font-medium">{formatDate(selectedTask.started || selectedTask.created)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-xs text-muted-foreground">Duration</span>
                        <span className="text-xs font-medium">{formatDuration(selectedTask.started, selectedTask.completed)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-xs text-muted-foreground">User</span>
                        <span className="text-xs font-medium">{selectedTask.user}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-xs text-muted-foreground">Project</span>
                        <span className="text-xs font-medium">{selectedTask.project_name}</span>
                      </div>
                    </div>
                  </div>

                  <div>
                    <h3 className="text-sm font-medium mb-2">Key Metrics</h3>
                    <div className="space-y-2">
                      {selectedTask.metrics && Object.entries(selectedTask.metrics).length > 0 ? (
                        Object.entries(selectedTask.metrics).slice(0, 6).map(([key, value]) => (
                          <div key={key} className="flex justify-between">
                            <span className="text-xs text-muted-foreground">{key}</span>
                            <span className="text-xs font-medium">
                              {typeof value === "number" ? value.toFixed(4) : value}
                            </span>
                          </div>
                        ))
                      ) : (
                        <p className="text-xs text-muted-foreground">No metrics available</p>
                      )}
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="text-sm font-medium mb-2">Tags</h3>
                  <div className="flex flex-wrap gap-2">
                    {selectedTask.tags && selectedTask.tags.length > 0 ? (
                      selectedTask.tags.map((tag, idx) => (
                        <Badge key={idx} variant="outline" className="text-xs">
                          {tag}
                        </Badge>
                      ))
                    ) : (
                      <p className="text-xs text-muted-foreground">No tags available</p>
                    )}
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="metrics">
                <div className="space-y-4">
                  <h3 className="text-sm font-medium">Performance Metrics</h3>
                  {selectedTask.metrics && Object.entries(selectedTask.metrics).length > 0 ? (
                    <div className="bg-muted/50 rounded-md p-4">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHeadCell>Metric</TableHeadCell>
                            <TableHeadCell>Value</TableHeadCell>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {Object.entries(selectedTask.metrics).map(([key, value]) => (
                            <TableRow key={key}>
                              <TableCell>{key}</TableCell>
                              <TableCell className="font-mono">
                                {typeof value === "number" ? value.toFixed(6) : String(value)}
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </div>
                  ) : (
                    <div className="bg-muted/50 rounded-md p-8 text-center">
                      <p className="text-muted-foreground">No metrics available for this task</p>
                    </div>
                  )}
                  
                  <div className="flex justify-center mt-4">
                    <p className="text-sm text-muted-foreground">
                      For detailed metric charts and plots, view this task in the
                      <Button variant="link" className="px-1 h-auto" onClick={() => window.open(selectedTask.web_url || '', '_blank')}>
                        ClearML UI
                      </Button>
                    </p>
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="params">
                <div className="max-h-[400px] overflow-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHeadCell>Parameter</TableHeadCell>
                        <TableHeadCell>Value</TableHeadCell>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {selectedTask.hyperparams && Object.keys(selectedTask.hyperparams).length > 0 ? (
                        Object.entries(selectedTask.hyperparams).map(([key, value]) => (
                          <TableRow key={key}>
                            <TableCell className="font-medium">{key}</TableCell>
                            <TableCell className="font-mono text-xs whitespace-pre-wrap break-all">
                              {typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)}
                            </TableCell>
                          </TableRow>
                        ))
                      ) : (
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
                  {selectedTask.artifacts && Object.keys(selectedTask.artifacts).length > 0 ? (
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHeadCell>Name</TableHeadCell>
                          <TableHeadCell>Type</TableHeadCell>
                          <TableHeadCell>Size</TableHeadCell>
                          <TableHeadCell className="text-right">Actions</TableHeadCell>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {Object.entries(selectedTask.artifacts).map(([name, details]) => (
                          <TableRow key={name}>
                            <TableCell className="font-medium">{name}</TableCell>
                            <TableCell>{(details as any).type || "Unknown"}</TableCell>
                            <TableCell>{(details as any).size || "Unknown"}</TableCell>
                            <TableCell className="text-right">
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => window.open(selectedTask.web_url || '', '_blank')}
                              >
                                <Eye className="h-4 w-4 mr-1" />
                                <span className="text-xs">View</span>
                              </Button>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  ) : (
                    <div className="bg-muted/50 rounded-md p-8 text-center">
                      <p className="text-muted-foreground">No artifacts available for this task</p>
                    </div>
                  )}
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
                <Button variant="outline" size="sm" onClick={() => window.open(selectedTask.web_url || '', '_blank')}>
                  <ExternalLink className="h-4 w-4 mr-1" />
                  Open in ClearML
                </Button>
                {selectedTask.status !== "completed" && selectedTask.status !== "failed" && (
                  <Button variant="default" size="sm">
                    <Play className="h-4 w-4 mr-1" />
                    Clone & Rerun
                  </Button>
                )}
              </div>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      )}
    </>
  )
}
