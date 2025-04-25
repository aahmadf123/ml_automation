"use client";

import { useState } from "react";
import { DashboardHeader } from "@/components/dashboard-header";
import { DashboardSidebar } from "@/components/dashboard-sidebar";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { 
  BarChart as BarChartIcon, 
  PieChart as PieChartIcon,
  LineChart as LineChartIcon,
  ScatterChart,
  Save,
  Download,
  RefreshCw,
  Plus,
  Copy,
  Share2,
  Settings,
  Filter,
  Database,
  FileText,
  Layers,
  X,
  Check,
  ExternalLink
} from "lucide-react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { 
  Dialog, 
  DialogContent, 
  DialogDescription, 
  DialogFooter, 
  DialogHeader, 
  DialogTitle 
} from "@/components/ui/dialog";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";

export default function VisualizationsPage() {
  const [activeTab, setActiveTab] = useState("builder");
  const [visualizationType, setVisualizationType] = useState("line");
  const [dataSource, setDataSource] = useState("prediction_metrics");
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [showShareDialog, setShowShareDialog] = useState(false);
  const [chartTitle, setChartTitle] = useState("Prediction Performance Trend");
  const [xAxis, setXAxis] = useState("date");
  const [yAxis, setYAxis] = useState("rmse");
  const [compareWith, setCompareWith] = useState("none");
  const [aggregation, setAggregation] = useState("none");
  const [dateRange, setDateRange] = useState("3months");
  const [includeBaseline, setIncludeBaseline] = useState(true);
  const [smoothing, setSmoothing] = useState([3]); // Slider value
  
  // Sample saved visualizations
  const savedVisualizations = [
    {
      id: "vis-001",
      title: "Monthly Loss Ratio Trends",
      type: "line",
      description: "Loss ratio trend analysis by month with 3-month moving average",
      created: "2023-03-15T10:45:00Z",
      lastModified: "2023-04-10T14:22:00Z",
      thumbnail: "/images/chart-thumbnails/loss-ratio-trend.png"
    },
    {
      id: "vis-002",
      title: "Claim Type Distribution",
      type: "pie",
      description: "Breakdown of claims by category and severity",
      created: "2023-02-20T09:30:00Z",
      lastModified: "2023-02-20T09:30:00Z",
      thumbnail: "/images/chart-thumbnails/claim-distribution.png"
    },
    {
      id: "vis-003",
      title: "Model Accuracy Comparison",
      type: "bar",
      description: "Side-by-side comparison of model accuracy metrics",
      created: "2023-04-05T16:15:00Z",
      lastModified: "2023-04-05T16:15:00Z",
      thumbnail: "/images/chart-thumbnails/model-accuracy.png"
    },
    {
      id: "vis-004",
      title: "Feature Correlation Matrix",
      type: "scatter",
      description: "Correlation analysis of key prediction features",
      created: "2023-03-28T11:20:00Z",
      lastModified: "2023-04-02T15:45:00Z",
      thumbnail: "/images/chart-thumbnails/feature-correlation.png"
    },
  ];
  
  // Sample data sources
  const dataSources = [
    { id: "prediction_metrics", name: "Prediction Performance Metrics", description: "Model prediction accuracy and performance over time" },
    { id: "loss_ratio", name: "Loss Ratio Data", description: "Historical and forecasted loss ratios by coverage type" },
    { id: "claims_data", name: "Claims Analysis", description: "Detailed claims data with category and severity breakdowns" },
    { id: "feature_importance", name: "Feature Importance", description: "Model feature importance rankings and drift analysis" },
    { id: "model_comparison", name: "Model Comparison", description: "Side-by-side comparison of model performance metrics" },
  ];
  
  // Helper function to get visualization type icon
  const getVisualizationTypeIcon = (type: string) => {
    switch (type) {
      case "line":
        return <LineChartIcon className="h-4 w-4" />;
      case "bar":
        return <BarChartIcon className="h-4 w-4" />;
      case "pie":
        return <PieChartIcon className="h-4 w-4" />;
      case "scatter":
        return <ScatterChart className="h-4 w-4" />;
      default:
        return <LineChartIcon className="h-4 w-4" />;
    }
  };
  
  // Format date for display
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString();
  };
  
  return (
    <div className="flex min-h-screen flex-col">
      <DashboardHeader title="Data Visualizations" />
      <div className="container flex-1 items-start md:grid md:grid-cols-[220px_minmax(0,1fr)] md:gap-6 lg:grid-cols-[240px_minmax(0,1fr)] lg:gap-10">
        <DashboardSidebar />
        <main className="flex w-full flex-col overflow-hidden p-4 md:p-6">
          {/* Top Actions */}
          <div className="flex items-center justify-between mb-6">
            <div>
              <h1 className="text-2xl font-bold">Data Visualizations</h1>
              <p className="text-muted-foreground">
                Create, customize, and share interactive data visualizations
              </p>
            </div>
            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm" onClick={() => setActiveTab("library")}>
                <FileText className="h-4 w-4 mr-2" />
                Saved Visualizations
              </Button>
              <Button size="sm" onClick={() => setActiveTab("builder")}>
                <Plus className="h-4 w-4 mr-2" />
                New Visualization
              </Button>
            </div>
          </div>
          
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="mb-4">
              <TabsTrigger value="builder" className="flex items-center gap-1">
                <Layers className="h-4 w-4" />
                Chart Builder
              </TabsTrigger>
              <TabsTrigger value="library" className="flex items-center gap-1">
                <FileText className="h-4 w-4" />
                Visualization Library
              </TabsTrigger>
              <TabsTrigger value="templates" className="flex items-center gap-1">
                <Copy className="h-4 w-4" />
                Templates
              </TabsTrigger>
            </TabsList>
            
            {/* Chart Builder Tab */}
            <TabsContent value="builder">
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left Panel - Chart Configuration */}
                <Card className="lg:col-span-1">
                  <CardHeader>
                    <CardTitle>Chart Configuration</CardTitle>
                    <CardDescription>
                      Configure your visualization
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {/* Visualization Type */}
                    <div className="space-y-2">
                      <Label>Visualization Type</Label>
                      <div className="grid grid-cols-4 gap-2">
                        <Button 
                          variant={visualizationType === "line" ? "default" : "outline"} 
                          className="h-20 flex flex-col items-center justify-center gap-1 py-2"
                          onClick={() => setVisualizationType("line")}
                        >
                          <LineChartIcon className="h-6 w-6" />
                          <span className="text-xs">Line</span>
                        </Button>
                        <Button 
                          variant={visualizationType === "bar" ? "default" : "outline"} 
                          className="h-20 flex flex-col items-center justify-center gap-1 py-2"
                          onClick={() => setVisualizationType("bar")}
                        >
                          <BarChartIcon className="h-6 w-6" />
                          <span className="text-xs">Bar</span>
                        </Button>
                        <Button 
                          variant={visualizationType === "pie" ? "default" : "outline"} 
                          className="h-20 flex flex-col items-center justify-center gap-1 py-2"
                          onClick={() => setVisualizationType("pie")}
                        >
                          <PieChartIcon className="h-6 w-6" />
                          <span className="text-xs">Pie</span>
                        </Button>
                        <Button 
                          variant={visualizationType === "scatter" ? "default" : "outline"} 
                          className="h-20 flex flex-col items-center justify-center gap-1 py-2"
                          onClick={() => setVisualizationType("scatter")}
                        >
                          <ScatterChart className="h-6 w-6" />
                          <span className="text-xs">Scatter</span>
                        </Button>
                      </div>
                    </div>
                    
                    {/* Data Source */}
                    <div className="space-y-2">
                      <Label>Data Source</Label>
                      <Select value={dataSource} onValueChange={setDataSource}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select data source" />
                        </SelectTrigger>
                        <SelectContent>
                          {dataSources.map(source => (
                            <SelectItem key={source.id} value={source.id}>
                              {source.name}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <p className="text-xs text-muted-foreground">
                        {dataSources.find(s => s.id === dataSource)?.description}
                      </p>
                    </div>
                    
                    {/* Chart Title */}
                    <div className="space-y-2">
                      <Label>Chart Title</Label>
                      <Input 
                        value={chartTitle} 
                        onChange={(e) => setChartTitle(e.target.value)} 
                        placeholder="Enter chart title"
                      />
                    </div>
                    
                    {/* Axis Configuration - Only show for line, bar, scatter */}
                    {(visualizationType === "line" || visualizationType === "bar" || visualizationType === "scatter") && (
                      <>
                        <div className="space-y-2">
                          <Label>X-Axis</Label>
                          <Select value={xAxis} onValueChange={setXAxis}>
                            <SelectTrigger>
                              <SelectValue placeholder="Select X-Axis" />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="date">Date</SelectItem>
                              <SelectItem value="category">Category</SelectItem>
                              <SelectItem value="region">Region</SelectItem>
                              <SelectItem value="model">Model</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                        
                        <div className="space-y-2">
                          <Label>Y-Axis</Label>
                          <Select value={yAxis} onValueChange={setYAxis}>
                            <SelectTrigger>
                              <SelectValue placeholder="Select Y-Axis" />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="rmse">RMSE</SelectItem>
                              <SelectItem value="mse">MSE</SelectItem>
                              <SelectItem value="mae">MAE</SelectItem>
                              <SelectItem value="r2">R² Score</SelectItem>
                              <SelectItem value="accuracy">Accuracy</SelectItem>
                              <SelectItem value="loss_ratio">Loss Ratio</SelectItem>
                              <SelectItem value="claim_amount">Claim Amount</SelectItem>
                              <SelectItem value="premium">Premium</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                      </>
                    )}
                    
                    {/* Compare With - Only show for line and bar */}
                    {(visualizationType === "line" || visualizationType === "bar") && (
                      <div className="space-y-2">
                        <Label>Compare With</Label>
                        <Select value={compareWith} onValueChange={setCompareWith}>
                          <SelectTrigger>
                            <SelectValue placeholder="Select comparison" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="none">No Comparison</SelectItem>
                            <SelectItem value="previous_period">Previous Period</SelectItem>
                            <SelectItem value="baseline">Baseline Model</SelectItem>
                            <SelectItem value="target">Target Threshold</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    )}
                    
                    {/* Time Range - Only show if X-Axis is date */}
                    {xAxis === "date" && (
                      <div className="space-y-2">
                        <Label>Time Range</Label>
                        <Select value={dateRange} onValueChange={setDateRange}>
                          <SelectTrigger>
                            <SelectValue placeholder="Select time range" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="1month">Last Month</SelectItem>
                            <SelectItem value="3months">Last 3 Months</SelectItem>
                            <SelectItem value="6months">Last 6 Months</SelectItem>
                            <SelectItem value="1year">Last Year</SelectItem>
                            <SelectItem value="ytd">Year to Date</SelectItem>
                            <SelectItem value="custom">Custom Range</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    )}
                    
                    {/* Aggregation */}
                    {/* Aggregation */}
                    <div className="space-y-2">
                      <Label>Data Aggregation</Label>
                      <Select value={aggregation} onValueChange={setAggregation}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select aggregation" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="none">None</SelectItem>
                          <SelectItem value="sum">Sum</SelectItem>
                          <SelectItem value="average">Average</SelectItem>
                          <SelectItem value="median">Median</SelectItem>
                          <SelectItem value="count">Count</SelectItem>
                          <SelectItem value="min">Minimum</SelectItem>
                          <SelectItem value="max">Maximum</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    
                    {/* Additional Settings - Line Chart Smoothing */}
                    {visualizationType === "line" && (
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <Label>Line Smoothing</Label>
                          <span className="text-xs text-muted-foreground">{smoothing[0]}</span>
                        </div>
                        <Slider 
                          value={smoothing} 
                          onValueChange={setSmoothing} 
                          min={0} 
                          max={10} 
                          step={1}
                        />
                      </div>
                    )}
                    
                    {/* Include Baseline */}
                    <div className="flex items-center justify-between space-y-0 pt-2">
                      <Label htmlFor="include-baseline">Include Baseline</Label>
                      <Switch 
                        id="include-baseline"
                        checked={includeBaseline}
                        onCheckedChange={setIncludeBaseline}
                      />
                    </div>
                  </CardContent>
                  <CardFooter className="border-t px-6 py-4">
                    <Button variant="outline" className="mr-2" onClick={() => setShowSaveDialog(true)}>
                      <Save className="h-4 w-4 mr-2" />
                      Save
                    </Button>
                    <Button variant="outline" className="mr-2" onClick={() => setShowShareDialog(true)}>
                      <Share2 className="h-4 w-4 mr-2" />
                      Share
                    </Button>
                    <Button>
                      <RefreshCw className="h-4 w-4 mr-2" />
                      Update Chart
                    </Button>
                  </CardFooter>
                </Card>
                
                {/* Right Panel - Chart Preview */}
                <Card className="lg:col-span-2">
                  <CardHeader className="flex flex-row items-center justify-between pb-2">
                    <div>
                      <CardTitle>{chartTitle || "Chart Preview"}</CardTitle>
                      <CardDescription>
                        Real-time preview of your visualization
                      </CardDescription>
                    </div>
                    <Button variant="outline" size="sm">
                      <Download className="h-4 w-4 mr-2" />
                      Export Chart
                    </Button>
                  </CardHeader>
                  <CardContent>
                    <div className="bg-card/50 border rounded-md flex items-center justify-center h-[500px]">
                      <div className="text-center p-6">
                        <div className="flex justify-center mb-4">
                          {getVisualizationTypeIcon(visualizationType)}
                        </div>
                        <p className="text-muted-foreground">
                          {visualizationType === "line" && "Line chart visualization will appear here"}
                          {visualizationType === "bar" && "Bar chart visualization will appear here"}
                          {visualizationType === "pie" && "Pie chart visualization will appear here"}
                          {visualizationType === "scatter" && "Scatter plot visualization will appear here"}
                        </p>
                        <p className="text-xs text-muted-foreground mt-2">
                          Showing data from: {dataSources.find(s => s.id === dataSource)?.name}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                  <CardFooter className="border-t flex justify-between px-6 py-4">
                    <div className="text-xs text-muted-foreground">
                      Last updated: {new Date().toLocaleString()}
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-muted-foreground">Auto-update</span>
                      <Switch defaultChecked id="auto-update" />
                    </div>
                  </CardFooter>
                </Card>
              </div>
            </TabsContent>
            
            {/* Visualization Library Tab */}
            <TabsContent value="library">
              <div className="flex justify-between items-center mb-6">
                <div className="relative flex-1 max-w-sm">
                  <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
                  <Input placeholder="Search visualizations..." className="pl-8" />
                </div>
                <div className="flex items-center gap-2">
                  <Select defaultValue="all">
                    <SelectTrigger className="w-36">
                      <SelectValue placeholder="Chart type" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Types</SelectItem>
                      <SelectItem value="line">Line Charts</SelectItem>
                      <SelectItem value="bar">Bar Charts</SelectItem>
                      <SelectItem value="pie">Pie Charts</SelectItem>
                      <SelectItem value="scatter">Scatter Plots</SelectItem>
                    </SelectContent>
                  </Select>
                  <Select defaultValue="date">
                    <SelectTrigger className="w-36">
                      <SelectValue placeholder="Sort by" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="date">Newest First</SelectItem>
                      <SelectItem value="name">Name (A-Z)</SelectItem>
                      <SelectItem value="type">Chart Type</SelectItem>
                      <SelectItem value="user">Created By</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-4">
                {savedVisualizations.map((vis) => (
                  <Card key={vis.id} className="overflow-hidden hover:shadow-md transition-shadow">
                    <div className="aspect-video bg-muted overflow-hidden">
                      <div className="bg-muted flex items-center justify-center h-full">
                        {getVisualizationTypeIcon(vis.type)}
                        <span className="ml-2">{vis.title}</span>
                      </div>
                    </div>
                    <CardContent className="p-4">
                      <h3 className="font-medium mb-1 flex items-center">
                        {vis.title}
                        <Badge variant="outline" className="ml-2 text-xs">
                          {vis.type}
                        </Badge>
                      </h3>
                      <p className="text-xs text-muted-foreground line-clamp-2 mb-2">
                        {vis.description}
                      </p>
                      <p className="text-xs text-muted-foreground mt-2">
                        Last modified: {formatDate(vis.lastModified)}
                      </p>
                    </CardContent>
                    <CardFooter className="p-4 pt-0 flex justify-between">
                      <Button variant="ghost" size="sm">
                        <Eye className="h-4 w-4 mr-2" />
                        View
                      </Button>
                      <Button variant="ghost" size="sm">
                        <Settings className="h-4 w-4 mr-2" />
                        Edit
                      </Button>
                    </CardFooter>
                  </Card>
                ))}
                
                {/* Create New Card */}
                <Card className="overflow-hidden border-dashed hover:shadow-md transition-shadow cursor-pointer" onClick={() => setActiveTab("builder")}>
                  <div className="aspect-video bg-muted/40 flex flex-col items-center justify-center">
                    <Plus className="h-8 w-8 mb-2 text-muted-foreground" />
                    <span className="text-sm text-muted-foreground">Create New</span>
                  </div>
                </Card>
              </div>
            </TabsContent>
            
            {/* Templates Tab */}
            <TabsContent value="templates">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Performance Over Time</CardTitle>
                    <CardDescription>
                      Line chart showing key metrics over time
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="pt-0">
                    <div className="aspect-video bg-muted rounded-md flex items-center justify-center">
                      <LineChartIcon className="h-8 w-8 text-muted-foreground" />
                    </div>
                    <div className="flex justify-end mt-4">
                      <Button size="sm" variant="default">
                        <Copy className="h-4 w-4 mr-2" />
                        Use Template
                      </Button>
                    </div>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Metric Comparison</CardTitle>
                    <CardDescription>
                      Bar chart comparing metrics across models
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="pt-0">
                    <div className="aspect-video bg-muted rounded-md flex items-center justify-center">
                      <BarChartIcon className="h-8 w-8 text-muted-foreground" />
                    </div>
                    <div className="flex justify-end mt-4">
                      <Button size="sm" variant="default">
                        <Copy className="h-4 w-4 mr-2" />
                        Use Template
                      </Button>
                    </div>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Category Distribution</CardTitle>
                    <CardDescription>
                      Pie chart showing distribution by category
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="pt-0">
                    <div className="aspect-video bg-muted rounded-md flex items-center justify-center">
                      <PieChartIcon className="h-8 w-8 text-muted-foreground" />
                    </div>
                    <div className="flex justify-end mt-4">
                      <Button size="sm" variant="default">
                        <Copy className="h-4 w-4 mr-2" />
                        Use Template
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
          </Tabs>
          
          {/* Save Visualization Dialog */}
          <Dialog open={showSaveDialog} onOpenChange={setShowSaveDialog}>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Save Visualization</DialogTitle>
                <DialogDescription>
                  Save your visualization for future use and sharing
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4 py-4">
                <div className="space-y-2">
                  <Label htmlFor="viz-name">Visualization Name</Label>
                  <Input id="viz-name" value={chartTitle} onChange={(e) => setChartTitle(e.target.value)} />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="viz-desc">Description</Label>
                  <Input id="viz-desc" placeholder="Brief description of this visualization" />
                </div>
                <div className="space-y-2">
                  <Label>Tags</Label>
                  <Input placeholder="Add tags separated by commas" />
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox id="make-public" />
                  <Label htmlFor="make-public">Make this visualization available to all team members</Label>
                </div>
              </div>
              <DialogFooter>
                <Button variant="outline" onClick={() => setShowSaveDialog(false)}>
                  Cancel
                </Button>
                <Button onClick={() => setShowSaveDialog(false)}>
                  Save Visualization
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
          
          {/* Share Visualization Dialog */}
          <Dialog open={showShareDialog} onOpenChange={setShowShareDialog}>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Share Visualization</DialogTitle>
                <DialogDescription>
                  Share this visualization with others
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4 py-4">
                <div className="p-4 bg-muted rounded-md">
                  <p className="text-sm mb-2 font-medium">Shareable Link</p>
                  <div className="flex">
                    <Input readOnly value="https://dashboard.example.com/visualizations/shared/abc123" />
                    <Button variant="ghost" size="sm" className="ml-2">
                      <Copy className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <Label>Share with specific people</Label>
                  <Input placeholder="Enter email addresses" />
                </div>
                
                <div className="space-y-3">
                  <Label>Permissions</Label>
                  <RadioGroup defaultValue="view">
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="view" id="r1" />
                      <Label htmlFor="r1">View only</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="edit" id="r2" />
                      <Label htmlFor="r2">Can edit</Label>
                    </div>
                  </RadioGroup>
                </div>
                
                <div className="flex items-center space-x-2">
                  <Checkbox id="send-notification" defaultChecked />
                  <Label htmlFor="send-notification">Send email notification</Label>
                </div>
              </div>
              <DialogFooter>
                <Button variant="outline" onClick={() => setShowShareDialog(false)}>
                  Cancel
                
