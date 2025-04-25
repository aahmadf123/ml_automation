"use client";

import { useState, useEffect } from "react";
import { DashboardHeader } from "@/components/dashboard-header";
import { DashboardSidebar } from "@/components/dashboard-sidebar";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { RecentMLflowRuns } from "@/components/recent-mlflow-runs";
import { ExternalLink, RefreshCw } from "lucide-react";

export default function MLflowIntegrationPage() {
  const [mlflowUrl, setMlflowUrl] = useState("http://3.146.46.179:5000");

  return (
    <div className="flex min-h-screen flex-col">
      <DashboardHeader title="MLflow Integration" />
      <div className="container flex-1 items-start md:grid md:grid-cols-[220px_minmax(0,1fr)] md:gap-6 lg:grid-cols-[240px_minmax(0,1fr)] lg:gap-10">
        <DashboardSidebar />
        <main className="flex w-full flex-col overflow-hidden p-4 md:p-6">
          <Card className="mb-6">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>MLflow Tracking Server</CardTitle>
                  <CardDescription>
                    Monitor experiments, runs, and model metrics
                  </CardDescription>
                </div>
                <div className="flex items-center gap-2">
                  <Button variant="outline" size="sm">
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Refresh
                  </Button>
                  <Button size="sm" asChild>
                    <a href={mlflowUrl} target="_blank" rel="noopener noreferrer">
                      <ExternalLink className="h-4 w-4 mr-2" />
                      Open MLflow UI
                    </a>
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="flex items-center p-4 bg-slate-50 dark:bg-slate-900 rounded-md mb-4">
                <div className="flex-1">
                  <h3 className="text-sm font-medium">Tracking Server URL</h3>
                  <p className="text-xs text-muted-foreground mt-1">{mlflowUrl}</p>
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-2 w-2 rounded-full bg-green-500"></div>
                  <span className="text-xs font-medium">Connected</span>
                </div>
              </div>
              
              <Tabs defaultValue="recent-runs">
                <TabsList className="mb-4">
                  <TabsTrigger value="recent-runs">Recent Runs</TabsTrigger>
                  <TabsTrigger value="experiments">Experiments</TabsTrigger>
                  <TabsTrigger value="models">Registered Models</TabsTrigger>
                </TabsList>
                
                <TabsContent value="recent-runs" className="space-y-4">
                  <RecentMLflowRuns />
                </TabsContent>
                
                <TabsContent value="experiments" className="space-y-4">
                  <Card>
                    <CardHeader>
                      <CardTitle>MLflow Experiments</CardTitle>
                      <CardDescription>
                        View and manage experiment tracking
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <p className="text-center py-6 text-muted-foreground">
                        Experiment management interface coming soon.
                      </p>
                    </CardContent>
                  </Card>
                </TabsContent>
                
                <TabsContent value="models" className="space-y-4">
                  <Card>
                    <CardHeader>
                      <CardTitle>Registered Models</CardTitle>
                      <CardDescription>
                        View and manage model registry
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <p className="text-center py-6 text-muted-foreground">
                        Model registry interface coming soon.
                      </p>
                    </CardContent>
                  </Card>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        </main>
      </div>
    </div>
  );
}

