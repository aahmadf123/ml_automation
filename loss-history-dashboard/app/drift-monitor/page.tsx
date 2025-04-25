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
  RefreshCw, 
  Download, 
  ChevronDown, 
  ArrowUpRight, 
  ArrowDownRight, 
  Activity,
  LineChart,
  BarChart,
  ScatterChart,
  AlertCircle,
  CheckCircle
} from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

export default function DriftMonitorPage() {
  const [selectedModel, setSelectedModel] = useState("model4");
  const [timeRange, setTimeRange] = useState("7days");
  const [featureSet, setFeatureSet] = useState("all");
  
  // Helper functions to get drift status - these would be replaced with actual data in a real implementation
  const getDriftStatus = () => {
    return { 
      hasDrift: true, 
      severity: "medium", 
      affectedFeatures: 3,
      driftScore: 0.28,
      lastDetected: "2 hours ago"
    };
  };
  
  const getAlertVariant = () => {
    const status = getDriftStatus();
    if (status.severity === "high") return "destructive";
    if (status.severity === "medium") return "default";
    return "default";
  };
  
  const getDriftStatusMessage = () => {
    const status = getDriftStatus();
    if (status.hasDrift) {
      return `${status.severity.charAt(0).toUpperCase() + status.severity.slice(1)} drift detected in ${status.affectedFeatures} features with a drift score of ${status.driftScore}. Last detected ${status.lastDetected}.`;
    }
    return "No significant drift detected in the current monitoring period.";
  };
  
  const getDriftSeverityBadge = () => {
    const status = getDriftStatus();
    if (status.severity === "high") return "destructive";
    if (status.severity === "medium") return "warning";
    return "outline";
  };
  
  // Mock data for feature stability
  const getFeatureStability = () => {
    return 72; // percentage
  };
  
  // Mock data for distribution score
  const getDistributionScore = () => {
    return 68; // percentage
  };
  
  // Mock data for performance impact
  const getPerformanceImpact = () => {
    return "-4.6%"; // percentage change
  };
  
  // Mock data for most drifted features
  const getMostDriftedFeatures = () => {
    return [
      { name: "customer_age", driftScore: 0.42, impact: "high" },
      { name: "claim_amount", driftScore: 0.37, impact: "high" },
      { name: "policy_tenure", driftScore: 0.29, impact: "medium" },
      { name: "zip_code", driftScore: 0.24, impact: "medium" },
      { name: "coverage_type", driftScore: 0.18, impact: "low" }
    ];
  };
  
  return (
    <div className="flex min-h-screen flex-col">
      <DashboardHeader title="Drift Monitor" />
      <div className="container flex-1 items-start md:grid md:grid-cols-[220px_minmax(0,1fr)] md:gap-6 lg:grid-cols-[240px_minmax(0,1fr)] lg:gap-10">
        <DashboardSidebar />
        <main className="flex w-full flex-col overflow-hidden p-4 md:p-6">
          {/* Drift Alert Banner */}
          <Alert className="mb-6" variant={getAlertVariant()}>
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Drift Status</AlertTitle>
            <AlertDescription>
              {getDriftStatusMessage()}
            </AlertDescription>
          </Alert>
          
          {/* Filter Controls */}
          <div className="flex flex-wrap items-center gap-4 mb-6">
            <Select value={selectedModel} onValueChange={setSelectedModel}>
              <SelectTrigger className="w-40">
                <SelectValue placeholder="Model" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="model1">Traditional Model</SelectItem>
                <SelectItem value="model4">Enhanced Model</SelectItem>
                <SelectItem value="model_ensemble">Ensemble Model</SelectItem>
              </SelectContent>
            </Select>
            
            <Select value={timeRange} onValueChange={setTimeRange}>
              <SelectTrigger className="w-36">
                <SelectValue placeholder="Time Range" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="24hours">Last 24 Hours</SelectItem>
                <SelectItem value="7days">Last 7 Days</SelectItem>
                <SelectItem value="30days">Last 30 Days</SelectItem>
                <SelectItem value="custom">Custom Range</SelectItem>
              </SelectContent>
            </Select>
            
            <Select value={featureSet} onValueChange={setFeatureSet}>
              <SelectTrigger className="w-40">
                <SelectValue placeholder="Feature Set" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Features</SelectItem>
                <SelectItem value="categorical">Categorical Features</SelectItem>
                <SelectItem value="numerical">Numerical Features</SelectItem>
                <SelectItem value="top10">Top 10 by Importance</SelectItem>
              </SelectContent>
            </Select>
            
            <div className="flex-1"></div>
            
            <Button variant="outline" size="sm">
              <RefreshCw className="h-4 w-4 mr-2" />
              Refresh
            </Button>
            <Button variant="outline" size="sm">
              <Download className="h-4 w-4 mr-2" />
              Export
            </Button>
          </div>
          
          {/* Drift Metrics Overview */}
          <Card className="mb-6">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Drift Detection Overview</CardTitle>
                  <CardDescription>
                    Key metrics for data drift and model health
                  </CardDescription>
                </div>
                <Badge variant={getDriftSeverityBadge()}>
                  {getDriftStatus().severity.toUpperCase()} DRIFT
                </Badge>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                <Card>
                  <CardContent className="p-4">
                    <div className="text-xs text-muted-foreground mb-2">Feature Stability</div>
                    <div className="text-2xl font-bold">{getFeatureStability()}%</div>
                    <Progress value={getFeatureStability()} className="mt-2" />
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="p-4">
                    <div className="text-xs text-muted-foreground mb-2">Distribution Score</div>
                    <div className="text-2xl font-bold">{getDistributionScore()}%</div>
                    <Progress value={getDistributionScore()} className="mt-2" />
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="p-4">
                    <div className="text-xs text-muted-foreground mb-2">Affected Features</div>
                    <div className="text-2xl font-bold">{getDriftStatus().affectedFeatures}</div>
                    <div className="text-xs text-amber-600 mt-1">3 with high impact</div>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="p-4">
                    <div className="text-xs text-muted-foreground mb-2">Performance Impact</div>
                    <div className="text-2xl font-bold flex items-center">
                      {getPerformanceImpact()}
                      <ArrowDownRight className="h-4 w-4 text-red-500 ml-2" />
                    </div>
                    <div className="text-xs text-red-600 mt-1">Model accuracy degrading</div>
                  </CardContent>
                </Card>
              </div>
              
              <Tabs defaultValue="overview">
                <TabsList>
                  <TabsTrigger value="overview">Drift Overview</TabsTrigger>
                  <TabsTrigger value="features">Feature Distribution</TabsTrigger>
                  <TabsTrigger value="trends">Historical Trends</TabsTrigger>
                  <TabsTrigger value="impact">Performance Impact</TabsTrigger>
                </TabsList>
                
                <TabsContent value="overview" className="pt-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-lg">Most Drifted Features</CardTitle>
                        <CardDescription>
                          Features with the highest drift scores
                        </CardDescription>
                      </CardHeader>
                      <CardContent className="pt-0">
                        <div className="space-y-4">
                          {getMostDriftedFeatures().map((feature, index) => (
                            <div key={index}>
                              <div className="flex justify-between mb-1">
                                <span className="text-sm font-medium flex items-center">
                                  {feature.name}
                                  {feature.impact === "high" && (
                                    <Badge variant="destructive" className="ml-2 text-xs">High Impact</Badge>
                                  )}
                                </span>
                                <span className="text-sm font-medium">Score: {feature.driftScore.toFixed(2)}</span>
                              </div>
                              <Progress 
                                value={feature.driftScore * 100} 
                                className={`h-2 ${
                                  feature.impact === "high" 
                                    ? "bg-red-100 [&>*]:bg-red-500" 
                                    : feature.impact === "medium" 
                                      ? "bg-amber-100 [&>*]:bg-amber-500"
                                      : "bg-blue-100 [&>*]:bg-blue-500"
                                }`}
                              />
                            </div>
                          ))}
                        </div>
                      </CardContent>
                    </Card>
                    
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-lg">Drift Summary</CardTitle>
                        <CardDescription>
                          Overall drift analysis
                        </CardDescription>
                      </CardHeader>
                      <CardContent className="pt-0">
                        <div className="h-60 flex items-center justify-center">
                          <p className="text-muted-foreground">Drift summary chart will be displayed here</p>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </TabsContent>
                
                <TabsContent value="features" className="pt-4">
                  <div className="grid grid-cols-1 gap-6">
                    <Card>
                      <CardHeader>
                        <CardTitle>Feature Distribution Comparison</CardTitle>
                        <CardDescription>
                          Current vs. baseline distribution for selected features
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="h-80 flex items-center justify-center">
                          <p className="text-muted-foreground">Feature distribution comparison charts will be displayed here</p>
                        </div>
                      </CardContent>
                    </Card>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <Card>
                        <CardHeader>
                          <CardTitle className="text-lg">Categorical Features</CardTitle>
                          <CardDescription>
                            Distribution change in categorical variables
                          </CardDescription>
                        </CardHeader>
                        <CardContent className="pt-0">
                          <div className="h-60 flex items-center justify-center">
                            <p className="text-muted-foreground">Categorical distribution chart will be displayed here</p>
                          </div>
                        </CardContent>
                      </Card>
                      
                      <Card>
                        <CardHeader>
                          <CardTitle className="text-lg">Numerical Features</CardTitle>
                          <CardDescription>
                            Distribution change in numerical variables
                          </CardDescription>
                        </CardHeader>
                        <CardContent className="pt-0">
                          <div className="h-60 flex items-center justify-center">
                            <p className="text-muted-foreground">Numerical distribution chart will be displayed here</p>
                          </div>
                        </CardContent>
                      </Card>
                    </div>
                  </div>
                </TabsContent>
                
                <TabsContent value="trends" className="pt-4">
                  <Card>
                    <CardHeader>
                      <CardTitle>Drift Trend Analysis</CardTitle>
                      <CardDescription>
                        Historical drift patterns over time
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="h-80 flex items-center justify-center">
                        <p className="text-muted-foreground">Drift trend chart will be displayed here</p>
                      </div>
                    </CardContent>
                  </Card>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-lg">Feature Importance Shift</CardTitle>
                        <CardDescription>
                          Changes in feature importance over time
                        </CardDescription>
                      </CardHeader>
                      <CardContent className="pt-0">
                        <div className="h-60 flex items-center justify-center">
                          <p className="text-muted-foreground">Feature importance shift chart will be displayed here</p>
                        </div>
                      </CardContent>
                    </Card>
                    
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-lg">Seasonal Patterns</CardTitle>

