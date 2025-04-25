"use client";

import { DashboardHeader } from "@/components/dashboard-header";
import { DashboardSidebar } from "@/components/dashboard-sidebar";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Download, BarChart, LineChart, RefreshCw } from "lucide-react";

export default function ModelMetricsPage() {
  return (
    <div className="flex min-h-screen flex-col">
      <DashboardHeader title="Model Metrics" />
      <div className="container flex-1 items-start md:grid md:grid-cols-[220px_minmax(0,1fr)] md:gap-6 lg:grid-cols-[240px_minmax(0,1fr)] lg:gap-10">
        <DashboardSidebar />
        <main className="flex w-full flex-col overflow-hidden p-4 md:p-6">
          <Card>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Model Metrics Dashboard</CardTitle>
                  <CardDescription>
                    Performance metrics across all models
                  </CardDescription>
                </div>
                <div className="flex items-center gap-2">
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
              <Tabs defaultValue="traditional">
                <TabsList>
                  <TabsTrigger value="traditional">Traditional Model</TabsTrigger>
                  <TabsTrigger value="enhanced">Enhanced Model</TabsTrigger>
                  <TabsTrigger value="all">All Models</TabsTrigger>
                </TabsList>
                
                <TabsContent value="traditional" className="pt-4">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                    <Card>
                      <CardContent className="p-4">
                        <div className="text-xs text-muted-foreground mb-2">RMSE</div>
                        <div className="text-2xl font-bold">0.0632</div>
                        <div className="text-xs text-green-600 mt-1">▼ 12% from baseline</div>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="p-4">
                        <div className="text-xs text-muted-foreground mb-2">MAE</div>
                        <div className="text-2xl font-bold">0.0487</div>
                        <div className="text-xs text-green-600 mt-1">▼ 8% from baseline</div>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="p-4">
                        <div className="text-xs text-muted-foreground mb-2">R² Score</div>
                        <div className="text-2xl font-bold">0.9512</div>
                        <div className="text-xs text-green-600 mt-1">▲ 6% from baseline</div>
                      </CardContent>
                    </Card>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-lg">Performance Over Time</CardTitle>
                      </CardHeader>
                      <CardContent className="pt-0">
                        <div className="h-80 flex items-center justify-center">
                          <p className="text-muted-foreground">Performance time series will be displayed here</p>
                        </div>
                      </CardContent>
                    </Card>
                    
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-lg">Error Distribution</CardTitle>
                      </CardHeader>
                      <CardContent className="pt-0">
                        <div className="h-80 flex items-center justify-center">
                          <p className="text-muted-foreground">Error distribution chart will be displayed here</p>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </TabsContent>
                
                <TabsContent value="enhanced" className="pt-4">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                    <Card>
                      <CardContent className="p-4">
                        <div className="text-xs text-muted-foreground mb-2

