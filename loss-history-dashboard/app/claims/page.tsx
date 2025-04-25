"use client";

import { useState } from "react";
import { DashboardHeader } from "@/components/dashboard-header";
import { DashboardSidebar } from "@/components/dashboard-sidebar";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { FileText, Download, RefreshCw, Search, Filter, BarChart, PieChart, TrendingUp, CalendarClock } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

export default function ClaimsPage() {
  const [dateRange, setDateRange] = useState("90days");
  
  return (
    <div className="flex min-h-screen flex-col">
      <DashboardHeader title="Claims Analysis" />
      <div className="container flex-1 items-start md:grid md:grid-cols-[220px_minmax(0,1fr)] md:gap-6 lg:grid-cols-[240px_minmax(0,1fr)] lg:gap-10">
        <DashboardSidebar />
        <main className="flex w-full flex-col overflow-hidden p-4 md:p-6">
          <Card className="mb-6">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Claims Analysis Dashboard</CardTitle>
                  <CardDescription>
                    Detailed claims analysis with cost breakdowns
                  </CardDescription>
                </div>
                <div className="flex items-center gap-2">
                  <Select value={dateRange} onValueChange={setDateRange}>
                    <SelectTrigger className="w-36">
                      <SelectValue placeholder="Date Range" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="30days">Last 30 Days</SelectItem>
                      <SelectItem value="90days">Last 90 Days</SelectItem>
                      <SelectItem value="6months">Last 6 Months</SelectItem>
                      <SelectItem value="1year">Last Year</SelectItem>
                    </SelectContent>
                  </Select>
                  <Button variant="outline" size="sm">
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Refresh
                  </Button>
                  <Button variant="outline" size="sm">
                    <Download className="h-4 w-4 mr-2" />
                    Export
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                <Card>
                  <CardContent className="p-4">
                    <div className="text-xs text-muted-foreground mb-2">Total Claims</div>
                    <div className="text-2xl font-bold">1,247</div>
                    <div className="text-xs text-amber-600 mt-1">+3.5% from previous period</div>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="p-4">
                    <div className="text-xs text-muted-foreground mb-2">Average Claim Cost</div>
                    <div className="text-2xl font-bold">$8,432</div>
                    <div className="text-xs text-red-600 mt-1">+5.2% from previous period</div>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="p-4">
                    <div className="text-xs text-muted-foreground mb-2">Open Claims</div>
                    <div className="text-2xl font-bold">312</div>
                    <div className="text-xs text-green-600 mt-1">-2.1% from previous period</div>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="p-4">
                    <div className="text-xs text-muted-foreground mb-2">Avg. Resolution Time</div>
                    <div className="text-2xl font-bold">27 days</div>
                    <div className="text-xs text-green-600 mt-1">-3.8% from previous period</div>
                  </CardContent>
                </Card>
              </div>
              
              <div className="flex items-center gap-4 mb-6">
                <div className="relative flex-1">
                  <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
                  <Input placeholder="Search claims..." className="pl-8" />
                </div>
                <Button variant="outline" className="gap-1">
                  <Filter className="h-4 w-4" />
                  Filters
                </Button>
              </div>
              
              <Tabs defaultValue="overview">
                <TabsList>
                  <TabsTrigger value="overview">Overview</TabsTrigger>
                  <TabsTrigger value="cost-analysis">Cost Analysis</TabsTrigger>
                  <TabsTrigger value="resolution">Resolution Metrics</TabsTrigger>
                  <TabsTrigger value="breakdown">Claims Breakdown</TabsTrigger>
                </TabsList>
                
                <TabsContent value="overview" className="pt-4">
                  <div className="grid grid-cols-1 gap-6">
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-lg">Claims Trend Analysis</CardTitle>
                        <CardDescription>
                          Historical claim volume and cost trends
                        </CardDescription>
                      </CardHeader>
                      <CardContent className="pt-0">
                        <div className="h-80 flex items-center justify-center">
                          <p className="text

