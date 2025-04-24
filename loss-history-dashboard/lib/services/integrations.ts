import { WebSocket } from 'ws';
import { getSecrets } from '../secrets';

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

interface ModelOverride {
  modelId: string;
  overrideReason: string;
  userId: string;
  approvedBy: string | null;
  metrics: Record<string, any>;
  notes: string;
  timestamp: string;
}

interface ApprovalDecision {
  taskId: string;
  runId: string;
  approved: boolean;
  approver: string;
  comment: string;
  timestamp: string;
}

export class IntegrationsService {
  private airflowApiUrl: string;
  private airflowUsername: string;
  private airflowPassword: string;
  private mlflowApiUrl: string;
  private slackToken: string;
  private clearmlApiUrl: string;
  private clearmlAccessKey: string;
  private clearmlSecretKey: string;
  private ws: WebSocket | null = null;
  private initialized: boolean = false;

  constructor() {
    // Default values that will be overridden during initialization
    this.airflowApiUrl = '';
    this.airflowUsername = '';
    this.airflowPassword = '';
    this.mlflowApiUrl = '';
    this.slackToken = '';
    this.clearmlApiUrl = '';
    this.clearmlAccessKey = '';
    this.clearmlSecretKey = '';
  }

  async initialize() {
    if (this.initialized) return;
    
    try {
      const secrets = await getSecrets();
      this.airflowApiUrl = secrets.AIRFLOW_API_URL;
      this.airflowUsername = secrets.AIRFLOW_USERNAME;
      this.airflowPassword = secrets.AIRFLOW_PASSWORD;
      this.mlflowApiUrl = secrets.MLFLOW_API_URL;
      this.slackToken = secrets.SLACK_TOKEN;
      this.clearmlApiUrl = secrets.CLEARML_API_URL || 'https://app.clear.ml/api';
      this.clearmlAccessKey = secrets.CLEARML_ACCESS_KEY || '';
      this.clearmlSecretKey = secrets.CLEARML_SECRET_KEY || '';
      this.initialized = true;
    } catch (error) {
      console.error('Failed to initialize IntegrationsService with secrets:', error);
      throw error;
    }
  }

  // Helper method to ensure initialization
  private async ensureInitialized() {
    if (!this.initialized) {
      await this.initialize();
    }
  }

  // ClearML Integration - Log model override
  async logOverrideToClearML(override: ModelOverride): Promise<string> {
    await this.ensureInitialized();
    
    if (!this.clearmlAccessKey || !this.clearmlSecretKey) {
      console.warn('ClearML credentials not configured');
      return 'mock-task-id';
    }
    
    try {
      // Create a new task in ClearML to log the override
      const createTaskResponse = await fetch(`${this.clearmlApiUrl}/tasks`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${Buffer.from(`${this.clearmlAccessKey}:${this.clearmlSecretKey}`).toString('base64')}`
        },
        body: JSON.stringify({
          name: `Model Override: ${override.modelId}`,
          type: 'data_processing',
          tags: ['override', 'manual', 'human-in-the-loop'],
          comment: override.overrideReason,
          project: 'Model Management'
        })
      });
      
      const taskData = await createTaskResponse.json();
      const taskId = taskData.data.id;
      
      // Add custom fields to the task
      await fetch(`${this.clearmlApiUrl}/tasks/${taskId}/add_or_update_custom_field`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${Buffer.from(`${this.clearmlAccessKey}:${this.clearmlSecretKey}`).toString('base64')}`
        },
        body: JSON.stringify({
          customFields: [
            {
              name: 'override_reason',
              value: override.overrideReason
            },
            {
              name: 'user_id',
              value: override.userId
            },
            {
              name: 'approved_by',
              value: override.approvedBy || 'N/A'
            },
            {
              name: 'model_id',
              value: override.modelId
            },
            {
              name: 'override_timestamp',
              value: override.timestamp
            },
            {
              name: 'notes',
              value: override.notes
            }
          ]
        })
      });
      
      // Log any metrics provided
      if (Object.keys(override.metrics).length > 0) {
        await fetch(`${this.clearmlApiUrl}/tasks/${taskId}/metrics`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${Buffer.from(`${this.clearmlAccessKey}:${this.clearmlSecretKey}`).toString('base64')}`
          },
          body: JSON.stringify({
            metrics: Object.entries(override.metrics).map(([key, value]) => ({
              name: key,
              value: value,
              timestamp: Date.now()
            }))
          })
        });
      }
      
      return taskId;
    } catch (error) {
      console.error('Error logging to ClearML:', error);
      // Return a mock ID so the process can continue even if ClearML logging fails
      return `error-mock-${Date.now()}`;
    }
  }

  // Airflow Integration
  async getDAGs(): Promise<AirflowDAG[]> {
    await this.ensureInitialized();
    
    try {
      const response = await fetch(`${this.airflowApiUrl}/api/v1/dags`, {
        headers: {
          'Authorization': `Basic ${Buffer.from(this.airflowUsername + ':' + this.airflowPassword).toString('base64')}`
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
    await this.ensureInitialized();
    
    try {
      const response = await fetch(`${this.airflowApiUrl}/api/v1/dags/${dagId}/dagRuns`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Basic ${Buffer.from(this.airflowUsername + ':' + this.airflowPassword).toString('base64')}`
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
    await this.ensureInitialized();
    
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
    await this.ensureInitialized();
    
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
    await this.ensureInitialized();
    
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
  async connectWebSocket() {
    await this.ensureInitialized();
    
    // The WebSocket endpoint might also come from secrets
    const wsEndpoint = process.env.WEBSOCKET_ENDPOINT || 'ws://localhost:8000/ws/metrics';
    
    this.ws = new WebSocket(wsEndpoint);

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

  // Set an Airflow variable
  async setAirflowVariable(name: string, value: string): Promise<boolean> {
    await this.ensureInitialized();
    
    if (!this.airflowApiUrl) {
      console.warn('Airflow API URL not configured');
      return false;
    }
    
    try {
      // Create a new variable in Airflow
      const response = await fetch(`${this.airflowApiUrl}/variables`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.airflowApiToken}`
        },
        body: JSON.stringify({
          key: name,
          value
        })
      });
      
      if (!response.ok) {
        // If variable already exists, try to update it
        if (response.status === 409) {
          const updateResponse = await fetch(`${this.airflowApiUrl}/variables/${name}`, {
            method: 'PATCH',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${this.airflowApiToken}`
            },
            body: JSON.stringify({
              value
            })
          });
          
          return updateResponse.ok;
        }
        return false;
      }
      
      return true;
    } catch (error) {
      console.error('Error setting Airflow variable:', error);
      return false;
    }
  }
  
  // ClearML Integration - Log approval decision
  async logApprovalToClearML(approval: ApprovalDecision): Promise<string> {
    await this.ensureInitialized();
    
    if (!this.clearmlAccessKey || !this.clearmlSecretKey) {
      console.warn('ClearML credentials not configured');
      return 'mock-task-id';
    }
    
    try {
      // Create a new task in ClearML to log the approval
      const createTaskResponse = await fetch(`${this.clearmlApiUrl}/tasks`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${Buffer.from(`${this.clearmlAccessKey}:${this.clearmlSecretKey}`).toString('base64')}`
        },
        body: JSON.stringify({
          name: `${approval.approved ? 'Approval' : 'Rejection'}: ${approval.taskId}`,
          type: 'data_processing',
          tags: ['approval', 'human-in-the-loop', approval.approved ? 'approved' : 'rejected'],
          comment: approval.comment || `${approval.approved ? 'Approved' : 'Rejected'} by ${approval.approver}`,
          project: 'Model Management'
        })
      });
      
      const taskData = await createTaskResponse.json();
      const taskId = taskData.data.id;
      
      // Add custom fields to the task
      await fetch(`${this.clearmlApiUrl}/tasks/${taskId}/add_or_update_custom_field`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${Buffer.from(`${this.clearmlAccessKey}:${this.clearmlSecretKey}`).toString('base64')}`
        },
        body: JSON.stringify({
          customFields: [
            {
              name: 'approval_decision',
              value: approval.approved ? 'approved' : 'rejected'
            },
            {
              name: 'approver',
              value: approval.approver
            },
            {
              name: 'task_id',
              value: approval.taskId
            },
            {
              name: 'run_id',
              value: approval.runId
            },
            {
              name: 'approval_timestamp',
              value: approval.timestamp
            },
            {
              name: 'comment',
              value: approval.comment || 'No comment provided'
            }
          ]
        })
      });
      
      return taskId;
    } catch (error) {
      console.error('Error logging to ClearML:', error);
      // Return a mock ID so the process can continue even if ClearML logging fails
      return `error-mock-${Date.now()}`;
    }
  }
} 