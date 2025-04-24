'use client';

import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { AirflowDAGs } from '@/components/airflow-dags';
import { MLflowExperiments } from '@/components/mlflow-experiments';
import { SlackNotifications } from '@/components/slack-notifications';
import { SystemMetrics } from '@/components/system-metrics';
import { DashboardLayout } from '@/components/dashboard-layout';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export default function IntegrationsPage() {
  return (
    <DashboardLayout
      title="Integrations"
      description="Manage your Airflow DAGs, MLflow experiments, and Slack notifications"
    >
      <div className="space-y-8">
        {/* Quick Stats */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Active DAGs</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">12</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">MLflow Experiments</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">8</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Model Accuracy</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">94.5%</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">System Health</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">98.2%</div>
            </CardContent>
          </Card>
        </div>

        {/* Main Content */}
        <Tabs defaultValue="airflow" className="space-y-4">
          <TabsList className="bg-black/20 backdrop-blur-sm">
            <TabsTrigger value="airflow">Airflow</TabsTrigger>
            <TabsTrigger value="mlflow">MLflow</TabsTrigger>
            <TabsTrigger value="slack">Slack</TabsTrigger>
            <TabsTrigger value="system">System</TabsTrigger>
          </TabsList>

          <TabsContent value="airflow" className="space-y-4">
            <AirflowDAGs />
          </TabsContent>

          <TabsContent value="mlflow" className="space-y-4">
            <MLflowExperiments />
          </TabsContent>

          <TabsContent value="slack" className="space-y-4">
            <SlackNotifications />
          </TabsContent>

          <TabsContent value="system" className="space-y-4">
            <SystemMetrics
              metrics={{
                runtime: 3600,
                memoryUsage: 1024 * 1024 * 100,
                cpuUsage: 45.5,
                gpuUsage: 80.0
              }}
            />
          </TabsContent>
        </Tabs>
      </div>
    </DashboardLayout>
  );
} 