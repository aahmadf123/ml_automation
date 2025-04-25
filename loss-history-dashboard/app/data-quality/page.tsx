"use client";

import { useState } from "react";
import { DashboardHeader } from "@/components/dashboard-header";
import { DashboardSidebar } from "@/components/dashboard-sidebar";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { CheckCircle, XCircle, AlertTriangle, FileWarning, PieChart, BarChart, CalendarDays, RefreshCw, Download, Filter, Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

export default function DataQualityPage() {
  const [dataSource, setDataSource] = useState("all");
  const [timeRange, setTimeRange] = useState("7days");

  return (
    <div className="flex min-h-screen flex-col">
      <DashboardHeader title="Data Quality" />
      <div className="container flex-1 items-start md:grid md:grid-cols-[220px_minmax(0,1fr)] md:gap-6 lg:grid-cols-[240px_minmax(0,1fr)] lg:gap-10">
        <DashboardSidebar />
        <main className="flex w-full flex-col overflow-hidden p-4 md:p-6">
          {/* Summary Alert */}
          <Alert className="mb-6" variant={getAlertVariant()}>
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Data Quality Status</AlertTitle>
            <AlertDescription>
              {getQualityStatusMessage()}
            </AlertDescription>
          </Alert>

          {/* Filter Controls */}
          <div className="flex items-center gap-4 mb-6">
            <Select value={dataSource} onValueChange={setDataSource}>
              <SelectTrigger className="w-40">
                <SelectValue placeholder="Data Source" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Sources</SelectItem>
                <SelectItem value="claims">Claims Data</SelectItem>
                <SelectItem value="customer">Customer Data</SelectItem>
                <SelectItem value="policy">Policy Data</SelectItem>
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
          
          {/* Data Quality Overview Card */}
          <Card className="mb-6">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Data Quality Overview</CardTitle>
                  <CardDescription>
                    Key data quality metrics and validation status
                  </CardDescription>
                </div>
                <Badge variant={getDataHealthBadgeVariant()}>
                  {getDataHealthStatus()}
                </Badge>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                <Card>
                  <CardContent className="p-4">
                    <div className="text-xs text-muted-foreground mb-2">Data Completeness</div>
                    <div className="text-2xl font-bold">{getCompleteness()}%</div>
                    <Progress value={getCompleteness()} className="mt-2" />
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="p-4">
                    <div className="text-xs text-muted-foreground mb-2">Validation Success</div>
                    <div className="text-2xl font-bold">{getValidationSuccess()}%</div>
                    <Progress value={getValidationSuccess()} className="mt-2" />
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="p-4">
                    <div className="text-xs text-muted-foreground mb-2">Critical Issues</div>
                    <div className="text-2xl font-bold flex items-center">
                      {getCriticalIssues()}
                      <Badge variant="destructive" className="ml-2">
                        {getCriticalIssuesTrend()}
                      </Badge>
                    </div>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="p-4">
                    <div className="text-xs text-muted-foreground mb-2">Schema Compliance</div>
                    <div className="text-2xl font-bold">{getSchemaCompliance()}%</div>
                    <Progress value={getSchemaCompliance()} className="mt-2" />
                  </CardContent>
                </Card>
              </div>
              
              <Tabs defaultValue="general">
                <TabsList>
                  <TabsTrigger value="general">General Metrics</TabsTrigger>
                  <TabsTrigger value="validation">Validation Details</TabsTrigger>
                  <TabsTrigger value="trends">Quality Trends</TabsTrigger>
                  <TabsTrigger value="issues">Issue Log</TabsTrigger>
                </TabsList>
                
                <TabsContent value="general" className="pt-4">
                  <div className="grid grid-cols-1 gap-6">
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-lg">Data Quality by Category</CardTitle>
                        <CardDescription>
                          Quality metrics across different data categories
                        </CardDescription>
                      </CardHeader>
                      <CardContent className="pt-0">
                        <div className="space-y-4">
                          <div>
                            <div className="flex justify-between mb-1">
                              <span className="text-sm font-medium">Customer Data</span>
                              <span className="text-sm font-medium">96.4%</span>
                            </div>
                            <Progress value={96.4} className="h-2" />
                          </div>
                          <div>
                            <div className="flex justify-between mb-1">
                              <span className="text-sm font-medium">Policy Data</span>
                              <span className="text-sm font-medium">98.7%</span>
                            </div>
                            <Progress value={98.7} className="h-2" />
                          </div>
                          <div>
                            <div className="flex justify-between mb-1">
                              <span className="text-sm font-medium">Claims Data</span>
                              <span className="text-sm font-medium">92.3%</span>
                            </div>
                            <Progress value={92.3} className="h-2" />
                          </div>
                          <div>
                            <div className="flex justify-between mb-1">
                              <span className="text-sm font-medium">Location Data</span>
                              <span className="text-sm font-medium">89.1%</span>
                            </div>
                            <Progress value={89.1} className="h-2" />
                          </div>
                          <div>
                            <div className="flex justify-between mb-1">
                              <span className="text-sm font-medium">Financial Data</span>
                              <span className="text-sm font-medium">97.8%</span>
                            </div>
                            <Progress value={97.8} className="h-2" />
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </TabsContent>
                
                <TabsContent value="validation" className="pt-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-lg">Validation Status</CardTitle>
                        <CardDescription>
                          Summary of data validation results
                        </CardDescription>
                      </CardHeader>
                      <CardContent className="pt-0">
                        <div className="h-60 flex items-center justify-center">
                          <p className="text-muted-foreground">Validation status chart will be displayed here</p>
                        </div>
                      </CardContent>
                    </Card>
                    
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-lg">Validation Details</CardTitle>
                        <CardDescription>
                          Details of recent validation runs
                        </CardDescription>
                      </CardHeader>
                      <CardContent className="pt-0">
                        <div className="space-y-4">
                          <div className="flex items-center p-2 border rounded-md bg-green-50">
                            <CheckCircle className="h-5 w-5 text-green-500 mr-2" />
                            <div>
                              <p className="text-sm font-medium">Schema Validation</p>
                              <p className="text-xs text-muted-foreground">All tables conform to expected schema</p>
                            </div>
                          </div>
                          <div className="flex items-center p-2 border rounded-md bg-red-50">
                            <XCircle className="h-5 w-5 text-red-500 mr-2" />
                            <div>
                              <p className="text-sm font-medium">Missing Values Check</p>
                              <p className="text-xs text-muted-foreground">3 columns have >5% missing values</p>
                            </div>
                          </div>
                          <div className="flex items-center p-2 border rounded-md bg-amber-50">
                            <AlertTriangle className="h-5 w-5 text-amber-500 mr-2" />
                            <div>
                              <p className="text-sm font-medium">Type Consistency</p>
                              <p className="text-xs text-muted-foreground">1 column has mixed data types</p>
                            </div>
                          </div>
                          <div className="flex items-center p-2 border rounded-md bg-green-50">
                            <CheckCircle className="h-5 w-5 text-green-500 mr-2" />
                            <div>
                              <p className="text-sm font-medium">Range Checks</p>
                              <p className="text-xs text-muted-foreground">All numeric fields within expected ranges</p>
                            </div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </TabsContent>
                
                <TabsContent value="trends" className="pt-4">
                  <Card>
                    <CardHeader>
                      <CardTitle>Data Quality Trends</CardTitle>
                      <CardDescription>
                        Quality metrics over time
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="h-80 flex items-center justify-center">
                        <p className="text-muted-foreground">Data quality trend chart will be displayed here</p>
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>
                
                <TabsContent value="issues" className="pt-4">
                  <Card>
                    <CardHeader>
                      <CardTitle>Issue Log</CardTitle>
                      <CardDescription>
                        Recent data quality issues
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        <div className="p-3 bg-red-50 border border-red-100 rounded-md">
                          <div className="flex items-center">
                            <FileWarning className="h-5 w-5 text-red-500 mr-2" />
                            <div className="flex-1">
                              <p className="text-sm font-medium">Missing Values in Customer Address</p>
                              <p className="text-xs text-muted-foreground">Detected 245 records with missing address data</p>
                            </div>
                            <Badge variant="outline" className="ml-2">Critical</Badge>
                          </div>
                          <div className="mt-2 text-xs text-muted-foreground">
                            <span className="font-medium">Impact:</span> May affect location-based risk assessment
                          </div>
                        </div>
                        
                        <div className="p-3 bg-amber-50 border border-amber-100 rounded-md">
                          <div className="flex items-center">
                            <AlertTriangle className="h-5 w-5 text-amber-500 mr-2" />
                            <div className="flex-1">
                              <p className="text-sm font-medium">Inconsistent Date Formats</p>
                              <p className="text-xs text-muted-foreground">Mixed date formats detected in policy data</p>
                            </div>
                            <Badge variant="outline" className="ml-2">Warning</Badge>
                          </div>
                          <div className="mt-2 text-xs text-muted-foreground">
                            <span className="font-medium">Impact:</span> May cause errors in date-based calculations
                          </div>
                        </div>
                        
                        <div className="p-3 bg-amber-50 border border-amber-100 rounded-md">
                          <div className="flex items-center">
                            <AlertTriangle className="h-5 w-5 text-amber-500 mr-2" />
                            <div className="flex-1">
                              <p className="text-sm font-medium">Outliers in Claim Amount</p>
                              <p className="text-xs text-muted-foreground">23 claim records with unusually high values</p>
                            </div>
                            <Badge variant="outline" className="ml-2">Warning</Badge>
                          </div>
                          <div className="mt-2 text-xs text-muted-foreground">
                            <span className="font-medium">Impact:</span> May skew average claim calculations
                          </div>
                        </div>
                      </div>
                    </CardContent>

