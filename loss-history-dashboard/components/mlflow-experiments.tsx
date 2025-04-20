import { useEffect, useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { IntegrationsService } from '@/lib/services/integrations';
import { useWebSocket } from '@/lib/websocket';
import { LineChart } from '@/components/ui/line-chart';

interface Experiment {
  experiment_id: string;
  name: string;
  artifact_location: string;
  lifecycle_stage: string;
  creation_time: number;
  last_update_time: number;
}

interface Run {
  run_id: string;
  status: string;
  start_time: number;
  end_time: number;
  metrics: Record<string, number>;
  params: Record<string, string>;
}

export function MLflowExperiments() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [selectedExperiment, setSelectedExperiment] = useState<string | null>(null);
  const [runs, setRuns] = useState<Run[]>([]);
  const [loading, setLoading] = useState(true);
  const { lastMessage } = useWebSocket();
  const integrationsService = new IntegrationsService();

  useEffect(() => {
    fetchExperiments();
  }, []);

  useEffect(() => {
    if (selectedExperiment) {
      fetchRuns(selectedExperiment);
    }
  }, [selectedExperiment]);

  useEffect(() => {
    if (lastMessage?.type === 'mlflow_update') {
      if (lastMessage.data.experiments) {
        setExperiments(lastMessage.data.experiments);
      }
      if (lastMessage.data.runs && lastMessage.data.experimentId === selectedExperiment) {
        setRuns(lastMessage.data.runs);
      }
    }
  }, [lastMessage, selectedExperiment]);

  const fetchExperiments = async () => {
    try {
      const fetchedExperiments = await integrationsService.getExperiments();
      setExperiments(fetchedExperiments);
    } catch (error) {
      console.error('Error fetching experiments:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchRuns = async (experimentId: string) => {
    try {
      const fetchedRuns = await integrationsService.getRuns(experimentId);
      setRuns(fetchedRuns);
    } catch (error) {
      console.error('Error fetching runs:', error);
    }
  };

  const formatDate = (timestamp: number) => {
    return new Date(timestamp).toLocaleString();
  };

  const formatMetricValue = (value: number) => {
    return value.toFixed(4);
  };

  if (loading) {
    return <div>Loading experiments...</div>;
  }

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-bold">MLflow Experiments</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {experiments.map((experiment) => (
          <Card
            key={experiment.experiment_id}
            className={`p-4 cursor-pointer ${
              selectedExperiment === experiment.experiment_id ? 'ring-2 ring-blue-500' : ''
            }`}
            onClick={() => setSelectedExperiment(experiment.experiment_id)}
          >
            <div className="space-y-2">
              <h3 className="text-lg font-semibold">{experiment.name}</h3>
              <div className="text-sm text-gray-500">
                <p>ID: {experiment.experiment_id}</p>
                <p>Stage: {experiment.lifecycle_stage}</p>
                <p>Created: {formatDate(experiment.creation_time)}</p>
                <p>Updated: {formatDate(experiment.last_update_time)}</p>
              </div>
            </div>
          </Card>
        ))}
      </div>

      {selectedExperiment && (
        <div className="mt-8">
          <h3 className="text-xl font-bold mb-4">Runs</h3>
          <div className="grid grid-cols-1 gap-4">
            {runs.map((run) => (
              <Card key={run.run_id} className="p-4">
                <div className="space-y-4">
                  <div className="flex justify-between items-start">
                    <div>
                      <h4 className="text-lg font-semibold">Run {run.run_id}</h4>
                      <p className="text-sm text-gray-500">
                        Status: {run.status}
                      </p>
                      <p className="text-sm text-gray-500">
                        Started: {formatDate(run.start_time)}
                      </p>
                      {run.end_time && (
                        <p className="text-sm text-gray-500">
                          Ended: {formatDate(run.end_time)}
                        </p>
                      )}
                    </div>
                    <Button
                      variant="outline"
                      onClick={() => window.open(`${process.env.MLFLOW_API_URL}/#/experiments/${selectedExperiment}/runs/${run.run_id}`, '_blank')}
                    >
                      View in MLflow
                    </Button>
                  </div>

                  {Object.entries(run.metrics).length > 0 && (
                    <div>
                      <h5 className="font-semibold mb-2">Metrics</h5>
                      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
                        {Object.entries(run.metrics).map(([key, value]) => (
                          <div key={key} className="text-sm">
                            <span className="font-medium">{key}:</span>{' '}
                            {formatMetricValue(value)}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {Object.entries(run.params).length > 0 && (
                    <div>
                      <h5 className="font-semibold mb-2">Parameters</h5>
                      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
                        {Object.entries(run.params).map(([key, value]) => (
                          <div key={key} className="text-sm">
                            <span className="font-medium">{key}:</span>{' '}
                            {value}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </Card>
            ))}
          </div>
        </div>
      )}
    </div>
  );
} 