import { WebSocket } from 'ws';

interface AirflowDAG {
  dag_id: string;
  is_paused: boolean;
  last_run: string;
  next_run: string;
  schedule: string;
}

interface MLflowRun {
  run_id: string;
  status: string;
  start_time: number;
  end_time: number;
  metrics: Record<string, number>;
  params: Record<string, string>;
}

interface SlackMessage {
  channel: string;
  text: string;
  blocks?: any[];
}

export class IntegrationsService {
  private airflowApiUrl: string;
  private mlflowApiUrl: string;
  private slackToken: string;
  private ws: WebSocket | null = null;

  constructor(
    airflowApiUrl: string = process.env.AIRFLOW_API_URL || 'http://localhost:8080',
    mlflowApiUrl: string = process.env.MLFLOW_API_URL || 'http://localhost:5000',
    slackToken: string = process.env.SLACK_TOKEN || ''
  ) {
    this.airflowApiUrl = airflowApiUrl;
    this.mlflowApiUrl = mlflowApiUrl;
    this.slackToken = slackToken;
  }

  // Airflow Integration
  async getDAGs(): Promise<AirflowDAG[]> {
    try {
      const response = await fetch(`${this.airflowApiUrl}/api/v1/dags`, {
        headers: {
          'Authorization': `Basic ${Buffer.from(process.env.AIRFLOW_USERNAME + ':' + process.env.AIRFLOW_PASSWORD).toString('base64')}`
        }
      });
      const data = await response.json();
      return data.dags;
    } catch (error) {
      console.error('Error fetching DAGs:', error);
      return [];
    }
  }

  async triggerDAG(dagId: string, conf?: Record<string, any>): Promise<boolean> {
    try {
      const response = await fetch(`${this.airflowApiUrl}/api/v1/dags/${dagId}/dagRuns`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Basic ${Buffer.from(process.env.AIRFLOW_USERNAME + ':' + process.env.AIRFLOW_PASSWORD).toString('base64')}`
        },
        body: JSON.stringify({ conf: conf || {} })
      });
      return response.ok;
    } catch (error) {
      console.error('Error triggering DAG:', error);
      return false;
    }
  }

  // MLflow Integration
  async getExperiments(): Promise<any[]> {
    try {
      const response = await fetch(`${this.mlflowApiUrl}/api/2.0/mlflow/experiments/list`);
      const data = await response.json();
      return data.experiments;
    } catch (error) {
      console.error('Error fetching experiments:', error);
      return [];
    }
  }

  async getRuns(experimentId: string): Promise<MLflowRun[]> {
    try {
      const response = await fetch(`${this.mlflowApiUrl}/api/2.0/mlflow/runs/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          experiment_ids: [experimentId],
          max_results: 100
        })
      });
      const data = await response.json();
      return data.runs;
    } catch (error) {
      console.error('Error fetching runs:', error);
      return [];
    }
  }

  // Slack Integration
  async sendSlackMessage(message: SlackMessage): Promise<boolean> {
    if (!this.slackToken) {
      console.warn('Slack token not configured');
      return false;
    }

    try {
      const response = await fetch('https://slack.com/api/chat.postMessage', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.slackToken}`
        },
        body: JSON.stringify(message)
      });
      const data = await response.json();
      return data.ok;
    } catch (error) {
      console.error('Error sending Slack message:', error);
      return false;
    }
  }

  // WebSocket connection for real-time updates
  connectWebSocket() {
    this.ws = new WebSocket('ws://localhost:8000/ws/metrics');

    this.ws.on('open', () => {
      console.log('Connected to WebSocket server');
    });

    this.ws.on('message', (data: string) => {
      try {
        const message = JSON.parse(data);
        this.handleWebSocketMessage(message);
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    });

    this.ws.on('error', (error) => {
      console.error('WebSocket error:', error);
    });

    this.ws.on('close', () => {
      console.log('WebSocket connection closed');
      // Attempt to reconnect after 5 seconds
      setTimeout(() => this.connectWebSocket(), 5000);
    });
  }

  private handleWebSocketMessage(message: any) {
    // Handle different types of messages
    switch (message.type) {
      case 'airflow_update':
        // Handle Airflow updates
        break;
      case 'mlflow_update':
        // Handle MLflow updates
        break;
      case 'system_metrics':
        // Handle system metrics updates
        break;
      default:
        console.warn('Unknown message type:', message.type);
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
} 