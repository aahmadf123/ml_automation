interface AirflowClientConfig {
  baseUrl: string
  username: string
  password: string
}

interface TriggerDagParams {
  dagId: string
  conf?: Record<string, any>
}

interface TriggerDagResponse {
  dagRunId: string
}

export class AirflowClient {
  private baseUrl: string
  private auth: string

  constructor(config: AirflowClientConfig) {
    this.baseUrl = config.baseUrl.replace(/\/$/, '')
    this.auth = Buffer.from(`${config.username}:${config.password}`).toString('base64')
  }

  // Static factory method to create client from secrets
  static async fromSecrets(): Promise<AirflowClient> {
    // For server-side use, import dynamically to avoid client-side bundling
    const { getSecrets } = await import('./secrets');
    const secrets = await getSecrets();
    
    return new AirflowClient({
      baseUrl: secrets.AIRFLOW_API_URL,
      username: secrets.AIRFLOW_USERNAME,
      password: secrets.AIRFLOW_PASSWORD
    });
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      ...options,
      headers: {
        'Authorization': `Basic ${this.auth}`,
        'Content-Type': 'application/json',
        ...options.headers,
      },
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({}))
      throw new Error(
        `Airflow API error: ${response.status} ${response.statusText} - ${JSON.stringify(error)}`
      )
    }

    return response.json()
  }

  async triggerDag(params: TriggerDagParams): Promise<TriggerDagResponse> {
    return this.request<TriggerDagResponse>('/api/v1/dags/trigger', {
      method: 'POST',
      body: JSON.stringify({
        conf: params.conf || {},
      }),
    })
  }

  async getDagRun(dagId: string, runId: string): Promise<any> {
    return this.request(`/api/v1/dags/${dagId}/dagRuns/${runId}`)
  }

  async getDagRuns(dagId: string): Promise<any> {
    return this.request(`/api/v1/dags/${dagId}/dagRuns`)
  }

  async getDagStatus(dagId: string): Promise<any> {
    return this.request(`/api/v1/dags/${dagId}`)
  }
} 