"use client";

import { useState } from "react";
import { DashboardHeader } from "@/components/dashboard-header";
import { DashboardSidebar } from "@/components/dashboard-sidebar";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { 
  AlertTriangle, 
  CheckCircle, 
  Clock, 
  RefreshCw, 
  Download, 
  Users, 
  BarChart, 
  CalendarDays,
  Bell,
  FileText,
  Eye,
  Workflow,
  History,
  Filter,
  Search,
  PlusCircle,
  XCircle
} from "lucide-react";
import { 
  Table, 
  TableBody, 
  TableCaption, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from "@/components/ui/table";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Progress } from "@/components/ui/progress";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

export default function IncidentsPage() {
  const [filterStatus, setFilterStatus] = useState("all");
  const [filterSeverity, setFilterSeverity] = useState("all");
  const [searchQuery, setSearchQuery] = useState("");
  
  // Sample incident data
  const activeIncidents = [
    {
      id: "INC-23094",
      title: "MLflow tracking server unresponsive",
      status: "active",
      severity: "high",
      createdAt: "2023-04-24T14:32:00Z",
      assignedTo: "Alex Kim",
      avatar: "/avatars/alex.png",
      systemsAffected: ["MLflow", "Model Registry"],
      description: "Users unable to log model metrics or access experiment tracking UI",
      updates: [
        { timestamp: "2023-04-24T14:32:00Z", message: "Incident created", author: "System" },
        { timestamp: "2023-04-24T14:35:12Z", message: "Investigating server logs", author: "Alex Kim" },
        { timestamp: "2023-04-24T14:47:35Z", message: "Identified memory leak in MLflow server", author: "Alex Kim" }
      ]
    },
    {
      id: "INC-23093",
      title: "Data pipeline processing delay",
      status: "investigating",
      severity: "medium",
      createdAt: "2023-04-24T12:15:00Z",
      assignedTo: "Jamie Chen",
      avatar: "/avatars/jamie.png",
      systemsAffected: ["Data Pipeline", "Feature Store"],
      description: "Data processing jobs taking 2.5x longer than normal, causing feature store refresh delays",
      updates: [
        { timestamp: "2023-04-24T12:15:00Z", message: "Incident created", author: "System" },
        { timestamp: "2023-04-24T12:18:43Z", message: "Investigating pipeline logs", author: "Jamie Chen" },
        { timestamp: "2023-04-24T12:42:21Z", message: "Found bottleneck in feature extraction step", author: "Jamie Chen" },
        { timestamp: "2023-04-24T13:15:08Z", message: "Applied temporary fix by scaling up workers", author: "Jamie Chen" }
      ]
    }
  ];
  
  const recentIncidents = [
    {
      id: "INC-23091",
      title: "Model prediction API latency",
      status: "resolved",
      severity: "medium",
      createdAt: "2023-04-23T08:45:00Z",
      resolvedAt: "2023-04-23T10:32:00Z",
      assignedTo: "Taylor Smith",
      avatar: "/avatars/taylor.png",
      systemsAffected: ["Prediction API", "Model Serving"],
      resolutionTime: "1h 47m",
      rootCause: "Insufficient memory allocation for model serving containers"
    },
    {
      id: "INC-23089",
      title: "Missing training data for daily retrain",
      status: "resolved",
      severity: "high",
      createdAt: "2023-04-22T05:15:00Z",
      resolvedAt: "2023-04-22T07:20:00Z",
      assignedTo: "Robin Lee",
      avatar: "/avatars/robin.png",
      systemsAffected: ["Data Ingest", "Training Pipeline"],
      resolutionTime: "2h 5m",
      rootCause: "Data source API authentication token expired"
    },
    {
      id: "INC-23085",
      title: "Dashboard access permission issue",
      status: "resolved",
      severity: "low",
      createdAt: "2023-04-20T13:10:00Z",
      resolvedAt: "2023-04-20T14:05:00Z",
      assignedTo: "Jordan Parker",
      avatar: "/avatars/jordan.png",
      systemsAffected: ["Dashboard UI", "Auth Service"],
      resolutionTime: "55m",
      rootCause: "Role permission misconfiguration after recent auth service update"
    }
  ];
  
  // Function to get incident severity badge variant
  const getSeverityBadge = (severity: string) => {
    switch (severity.toLowerCase()) {
      case "critical":
        return "destructive";
      case "high":
        return "destructive";
      case "medium":
        return "warning";
      case "low":
        return "secondary";
      default:
        return "outline";
    }
  };
  
  // Function to get incident status badge variant
  const getStatusBadge = (status: string) => {
    switch (status.toLowerCase()) {
      case "active":
        return "destructive";
      case "investigating":
        return "warning";
      case "mitigated":
        return "outline";
      case "resolved":
        return "success";
      default:
        return "outline";
    }
  };
  
  // Function to format date
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleString();
  };
  
  return (
    <div className="flex min-h-screen flex-col">
      <DashboardHeader title="System Incidents" />
      <div className="container flex-1 items-start md:grid md:grid-cols-[220px_minmax(0,1fr)] md:gap-6 lg:grid-cols-[240px_minmax(0,1fr)] lg:gap-10">
        <DashboardSidebar />
        <main className="flex w-full flex-col overflow-hidden p-4 md:p-6">
          {/* Active Incidents Alert */}
          {activeIncidents.length > 0 && (
            <Alert className="mb-6" variant="destructive">
              <AlertTriangle className="h-4 w-4" />
              <AlertTitle>Active Incidents</AlertTitle>
              <AlertDescription>
                There are {activeIncidents.length} active incidents requiring attention. Check the incident dashboard for details.
              </AlertDescription>
            </Alert>
          )}
          
          {/* Controls */}
          <div className="flex flex-wrap items-center gap-4 mb-6">
            <div className="relative flex-1">
              <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input 
                placeholder="Search incidents..." 
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-8"
              />
            </div>
            
            <Select value={filterStatus} onValueChange={setFilterStatus}>
              <SelectTrigger className="w-32">
                <SelectValue placeholder="Status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Statuses</SelectItem>
                <SelectItem value="active">Active</SelectItem>
                <SelectItem value="investigating">Investigating</SelectItem>
                <SelectItem value="mitigated">Mitigated</SelectItem>
                <SelectItem value="resolved">Resolved</SelectItem>
              </SelectContent>
            </Select>
            
            <Select value={filterSeverity} onValueChange={setFilterSeverity}>
              <SelectTrigger className="w-32">
                <SelectValue placeholder="Severity" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Severities</SelectItem>
                <SelectItem value="critical">Critical</SelectItem>
                <SelectItem value="high">High</SelectItem>
                <SelectItem value="medium">Medium</SelectItem>
                <SelectItem value="low">Low</SelectItem>
              </SelectContent>
            </Select>
            
            <Button variant="outline" size="sm">
              <RefreshCw className="h-4 w-4 mr-2" />
              Refresh
            </Button>
            
            <Button>
              <PlusCircle className="h-4 w-4 mr-2" />
              New Incident
            </Button>
          </div>
          
          {/* Incident Overview */}
          <Card className="mb-6">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Incident Overview</CardTitle>
                  <CardDescription>
                    Track and manage system incidents
                  </CardDescription>
                </div>
                <Badge 
                  variant={activeIncidents.length > 0 ? "destructive" : "success"} 
                  className="px-3 py-1"
                >
                  {activeIncidents.length === 0 ? "All Systems Operational" : `${activeIncidents.length} Active Incidents`}
                </Badge>
              </div>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="active">
                <TabsList>
                  <TabsTrigger value="active" className="flex items-center gap-1">
                    <AlertTriangle className="h-4 w-4" />
                    Active Incidents
                  </TabsTrigger>
                  <TabsTrigger value="recent" className="flex items-center gap-1">
                    <History className="h-4 w-4" />
                    Recent Incidents
                  </TabsTrigger>
                  <TabsTrigger value="stats" className="flex items-center gap-1">
                    <BarChart className="h-4 w-4" />
                    Statistics
                  </TabsTrigger>
                  <TabsTrigger value="oncall" className="flex items-center gap-1">
                    <Users className="h-4 w-4" />
                    On-Call Schedule
                  </TabsTrigger>
                </TabsList>
                
                {/* Active Incidents Tab */}
                <TabsContent value="active" className="pt-4">
                  {activeIncidents.length === 0 ? (
                    <div className="flex flex-col items-center justify-center p-6 text-center">
                      <CheckCircle className="h-12 w-12 text-green-500 mb-4" />
                      <h3 className="text-lg font-medium mb-2">All Systems Operational</h3>
                      <p className="text-muted-foreground">There are no active incidents at this time.</p>
                    </div>
                  ) : (
                    <div className="space-y-6">
                      {activeIncidents.map((incident, index) => (
                        <Card key={index} className="border-l-4" style={{ borderLeftColor: incident.severity === 'high' ? 'var(--destructive)' : 'var(--warning)' }}>
                          <CardHeader className="pb-2">
                            <div className="flex items-start justify-between">
                              <div>
                                <CardTitle className="text-lg">{incident.id}: {incident.title}</CardTitle>
                                <CardDescription>
                                  Reported on {formatDate(incident.createdAt)}
                                </CardDescription>
                              </div>
                              <div className="flex items-center gap-2">
                                <Badge variant={getSeverityBadge(incident.severity)}>
                                  {incident.severity.toUpperCase()}
                                </Badge>
                                <Badge variant={getStatusBadge(incident.status)}>
                                  {incident.status.toUpperCase()}
                                </Badge>
                              </div>
                            </div>
                          </CardHeader>
                          <CardContent className="pb-1">
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                              <div>
                                <h4 className="text-sm font-medium mb-1">Assigned To</h4>
                                <div className="flex items-center">
                                  <Avatar className="h-6 w-6 mr-2">
                                    <AvatarFallback>{incident.assignedTo.split(' ').map(n => n[0]).join('')}</AvatarFallback>
                                  </Avatar>
                                  <span className="text-sm">{incident.assignedTo}</span>
                                </div>
                              </div>
                              <div>
                                <h4 className="text-sm font-medium mb-1">Systems Affected</h4>
                                <div className="flex flex-wrap gap-1">
                                  {incident.systemsAffected.map((system, i) => (
                                    <Badge key={i} variant="outline" className="text-xs">
                                      {system}
                                    </Badge>
                                  ))}
                                </div>
                              </div>
                              <div>
                                <h4 className="text-sm font-medium mb-1">Duration</h4>
                                <div className="flex items-center">
                                  <Clock className="h-4 w-4 mr-1 text-muted-foreground" />
                                  <span className="text-sm">
                                    {calculateDuration(incident.createdAt)}
                                  </span>
                                </div>
                              </div>
                            </div>
                            <div className="mb-3">
                              <h4 className="text-sm font-medium mb-1">Description</h4>
                              <p className="text-sm text-muted-foreground">{incident.description}</p>
                            </div>
                            <div>
                              <h4 className="text-sm font-medium mb-1">Recent Updates</h4>
                              <div className="space-y-2">
                                {incident.updates.slice(-2).map((update, i) => (
                                  <div key={i} className="text-xs p-2 bg-muted rounded-md">
                                    <div className="flex justify-between mb-1">
                                      <span className="font-medium">{update.author}</span>
                

