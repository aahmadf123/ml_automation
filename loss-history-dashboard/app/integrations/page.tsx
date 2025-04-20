'use client';

import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { AirflowDAGs } from '@/components/airflow-dags';
import { MLflowExperiments } from '@/components/mlflow-experiments';
import { SlackNotifications } from '@/components/slack-notifications';
import { SystemMetrics } from '@/components/system-metrics';
import { DashboardLayout } from '@/components/dashboard-layout';
import { HolographicCard } from '@/components/ui/holographic-card';
import { Metrics3DVisualization } from '@/components/3d-metrics-visualization';
import { NeuralNetworkViz } from '@/components/neural-network-viz';

export default function IntegrationsPage() {
  // Example data for 3D visualization
  const metricsData = [
    { x: 0, y: 0, z: 0, value: 1, color: '#60a5fa' },
    { x: 1, y: 1, z: 1, value: 0.8, color: '#34d399' },
    { x: -1, y: -1, z: -1, value: 0.6, color: '#f472b6' },
    // Add more points as needed
  ];

  // Example neural network architecture
  const networkLayers = [
    { neurons: 4, activation: 'ReLU' },
    { neurons: 6, activation: 'ReLU' },
    { neurons: 6, activation: 'ReLU' },
    { neurons: 2, activation: 'Softmax' },
  ];

  return (
    <DashboardLayout
      title="Integrations"
      description="Manage your Airflow DAGs, MLflow experiments, and Slack notifications"
    >
      <div className="space-y-8">
        {/* Quick Stats */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <HolographicCard
            title="Active DAGs"
            value={12}
            gradient="from-blue-500/20 via-cyan-500/20 to-teal-500/20"
          />
          <HolographicCard
            title="MLflow Experiments"
            value={8}
            gradient="from-purple-500/20 via-pink-500/20 to-rose-500/20"
          />
          <HolographicCard
            title="Model Accuracy"
            value="94.5"
            unit="%"
            gradient="from-green-500/20 via-emerald-500/20 to-teal-500/20"
          />
          <HolographicCard
            title="System Health"
            value="98.2"
            unit="%"
            gradient="from-yellow-500/20 via-amber-500/20 to-orange-500/20"
          />
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
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <AirflowDAGs />
              <div className="space-y-4">
                <NeuralNetworkViz
                  layers={networkLayers}
                  height={400}
                />
                <Metrics3DVisualization
                  data={metricsData}
                  height={300}
                />
              </div>
            </div>
          </TabsContent>

          <TabsContent value="mlflow" className="space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <MLflowExperiments />
              <div className="space-y-4">
                <Metrics3DVisualization
                  data={metricsData}
                  height={600}
                />
              </div>
            </div>
          </TabsContent>

          <TabsContent value="slack" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <SlackNotifications />
              <div className="space-y-4">
                <NeuralNetworkViz
                  layers={networkLayers}
                  height={400}
                />
              </div>
            </div>
          </TabsContent>

          <TabsContent value="system" className="space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <SystemMetrics
                metrics={{
                  runtime: 3600,
                  memoryUsage: 1024 * 1024 * 100,
                  cpuUsage: 45.5,
                  gpuUsage: 80.0
                }}
              />
              <div className="space-y-4">
                <Metrics3DVisualization
                  data={metricsData}
                  height={400}
                />
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </DashboardLayout>
  );
} 