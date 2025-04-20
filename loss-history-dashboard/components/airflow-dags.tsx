import { useEffect, useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { IntegrationsService } from '@/lib/services/integrations';
import { useWebSocket } from '@/lib/websocket';

interface DAG {
  dag_id: string;
  is_paused: boolean;
  last_run: string;
  next_run: string;
  schedule: string;
}

export function AirflowDAGs() {
  const [dags, setDags] = useState<DAG[]>([]);
  const [loading, setLoading] = useState(true);
  const { lastMessage } = useWebSocket();
  const integrationsService = new IntegrationsService();

  useEffect(() => {
    fetchDAGs();
  }, []);

  useEffect(() => {
    if (lastMessage?.type === 'airflow_update') {
      setDags(lastMessage.data.dags);
    }
  }, [lastMessage]);

  const fetchDAGs = async () => {
    try {
      const fetchedDAGs = await integrationsService.getDAGs();
      setDags(fetchedDAGs);
    } catch (error) {
      console.error('Error fetching DAGs:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleTriggerDAG = async (dagId: string) => {
    try {
      const success = await integrationsService.triggerDAG(dagId);
      if (success) {
        // Refresh DAGs list
        fetchDAGs();
      }
    } catch (error) {
      console.error('Error triggering DAG:', error);
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  if (loading) {
    return <div>Loading DAGs...</div>;
  }

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-bold">Airflow DAGs</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {dags.map((dag) => (
          <Card key={dag.dag_id} className="p-4">
            <div className="space-y-2">
              <h3 className="text-lg font-semibold">{dag.dag_id}</h3>
              <div className="text-sm text-gray-500">
                <p>Status: {dag.is_paused ? 'Paused' : 'Active'}</p>
                <p>Last Run: {formatDate(dag.last_run)}</p>
                <p>Next Run: {formatDate(dag.next_run)}</p>
                <p>Schedule: {dag.schedule}</p>
              </div>
              <Button
                onClick={() => handleTriggerDAG(dag.dag_id)}
                disabled={dag.is_paused}
              >
                Trigger DAG
              </Button>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
} 