/**
 * ClearML Integration Service
 * 
 * Provides functionality to interact with ClearML for:
 * - Experiment tracking
 * - Pipeline monitoring
 * - Model metrics retrieval
 * - Task status monitoring
 */

import { getSecrets } from '../secrets';

// Define interfaces for ClearML responses
interface ClearMLExperiment {
  id: string;
  name: string;
  description?: string;
  tags?: string[];
  created: string;
  last_update: string;
}

interface ClearMLTask {
  id: string;
  name: string;
  type: string;
  status: string;
  created: string;
  last_update: string;
  started: string;
  completed?: string;
  project_id: string;
  project_name: string;
  user: string;
  tags?: string[];
  system_tags?: string[];
  script?: string;
  hyperparams?: Record<string, any>;
  configuration?: Record<string, any>;
  metrics?: Record<string, any>;
  artifacts?: Record<string, any>;
  models?: any[];
}

interface ClearMLMetric {
  task: string;
  metric: string;
  variant: string;
  value: number;
  timestamp: number;
  step?: number;
}

interface ClearMLModel {
  id: string;
  name: string;
  task_id: string;
  framework: string;
  uri: string;
  tags?: string[];
  created: string;
  labels?: Record<string, any>;
  design?: Record<string, any>;
  user?: string;
}

interface ClearMLPipeline {
  id: string;
  name: string;
  project_id: string;
  status: string;
  created: string;
  started?: string;
  completed?: string;
  nodes: ClearMLPipelineNode[];
}

interface ClearMLPipelineNode {
  id: string;
  name: string;
  task_id?: string;
  status: string;
  started?: string;
  completed?: string;
  job_id?: string;
}

export class ClearMLService {
  private apiUrl: string;
  private webUrl: string;
  private filesUrl: string;
  private accessKey: string;
  private secretKey: string;
  private authHeader: string;
  private initialized: boolean = false;

  constructor() {
    // Initialize with default values that will be overridden by secrets
    this.apiUrl = '';
    this.webUrl = '';
    this.filesUrl = '';
    this.accessKey = '';
    this.secretKey = '';
    this.authHeader = '';
  }

  /**
   * Initialize the ClearML service with credentials
   */
  async initialize() {
    if (this.initialized) return;
    
    try {
      const secrets = await getSecrets();
      this.apiUrl = secrets.CLEARML_API_HOST || 'https://api.clear.ml';
      this.webUrl = secrets.CLEARML_WEB_HOST || 'https://app.clear.ml';
      this.filesUrl = secrets.CLEARML_FILES_HOST || 'https://files.clear.ml';
      this.accessKey = secrets.CLEARML_API_ACCESS_KEY || '';
      this.secretKey = secrets.CLEARML_SECRET_KEY || '';
      
      // Generate auth header
      this.authHeader = `Bearer ${Buffer.from(`${this.accessKey}:${this.secretKey}`).toString('base64')}`;
      this.initialized = true;
      
      console.log('ClearML service initialized successfully');
    } catch (error) {
      console.error('Failed to initialize ClearML service:', error);
      throw new Error('Failed to initialize ClearML service');
    }
  }

  /**
   * Ensure the service is initialized before making API calls
   */
  private async ensureInitialized() {
    if (!this.initialized) {
      await this.initialize();
    }
  }

  /**
   * Get all projects from ClearML
   */
  async getProjects(): Promise<any[]> {
    await this.ensureInitialized();
    
    try {
      const response = await fetch(`${this.apiUrl}/projects`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': this.authHeader
        }
      });
      
      if (!response.ok) {
        throw new Error(`Failed to get projects: ${response.statusText}`);
      }
      
      const data = await response.json();
      return data.data || [];
    } catch (error) {
      console.error('Error fetching ClearML projects:', error);
      return [];
    }
  }

  /**
   * Get all experiments (projects) from ClearML
   * This provides MLflow-compatible interface
   */
  async getExperiments(): Promise<ClearMLExperiment[]> {
    const projects = await this.getProjects();
    
    return projects.map(project => ({
      id: project.id,
      name: project.name,
      description: project.description,
      tags: project.tags,
      created: project.created,
      last_update: project.last_updated
    }));
  }

  /**
   * Get tasks for a specific project
   */
  async getTasks(projectId: string, statusFilter?: string[]): Promise<ClearMLTask[]> {
    await this.ensureInitialized();
    
    try {
      let url = `${this.apiUrl}/tasks?project_id=${projectId}`;
      
      if (statusFilter && statusFilter.length > 0) {
        url += `&status=${statusFilter.join(',')}`;
      }
      
      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': this.authHeader
        }
      });
      
      if (!response.ok) {
        throw new Error(`Failed to get tasks: ${response.statusText}`);
      }
      
      const data = await response.json();
      return data.data?.tasks || [];
    } catch (error) {
      console.error('Error fetching ClearML tasks:', error);
      return [];
    }
  }

  /**
   * Get experiment runs (tasks) from ClearML
   * This provides MLflow-compatible interface
   */
  async getRuns(experimentId: string): Promise<any[]> {
    const tasks = await this.getTasks(experimentId);
    
    return tasks.map(task => ({
      run_id: task.id,
      experiment_id: task.project_id,
      status: this.mapTaskStatus(task.status),
      start_time: task.started,
      end_time: task.completed,
      metrics: task.metrics || {},
      params: task.hyperparams || {},
      tags: task.tags || [],
      artifacts: task.artifacts || {},
    }));
  }

  /**
   * Map ClearML task status to MLflow status
   */
  private mapTaskStatus(clearmlStatus: string): string {
    const statusMap: Record<string, string> = {
      'created': 'SCHEDULED',
      'queued': 'SCHEDULED',
      'in_progress': 'RUNNING',
      'stopped': 'FINISHED',
      'completed': 'FINISHED',
      'published': 'FINISHED',
      'publishing': 'RUNNING',
      'failed': 'FAILED',
      'unknown': 'UNKNOWN'
    };
    
    return statusMap[clearmlStatus] || 'UNKNOWN';
  }

  /**
   * Get a specific task by ID
   */
  async getTask(taskId: string): Promise<ClearMLTask | null> {
    await this.ensureInitialized();
    
    try {
      const response = await fetch(`${this.apiUrl}/tasks/${taskId}`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': this.authHeader
        }
      });
      
      if (!response.ok) {
        throw new Error(`Failed to get task: ${response.statusText}`);
      }
      
      const data = await response.json();
      return data.data || null;
    } catch (error) {
      console.error(`Error fetching ClearML task ${taskId}:`, error);
      return null;
    }
  }

  /**
   * Get task metrics
   */
  async getTaskMetrics(taskId: string): Promise<Record<string, any>> {
    await this.ensureInitialized();
    
    try {
      const response = await fetch(`${this.apiUrl}/tasks/${taskId}/metrics`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': this.authHeader
        }
      });
      
      if (!response.ok) {
        throw new Error(`Failed to get task metrics: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      // Format metrics in MLflow-compatible format
      const formattedMetrics: Record<string, any> = {};
      
      Object.entries(data.data || {}).forEach(([metricName, variants]) => {
        if (typeof variants === 'object') {
          Object.entries(variants).forEach(([variantName, values]) => {
            const key = variantName === 'value' ? metricName : `${metricName}/${variantName}`;
            // Use the last value as the current metric value
            if (Array.isArray(values) && values.length > 0) {
              formattedMetrics[key] = values[values.length - 1].value;
            }
          });
        }
      });
      
      return formattedMetrics;
    } catch (error) {
      console.error(`Error fetching metrics for task ${taskId}:`, error);
      return {};
    }
  }

  /**
   * Get models from ClearML
   */
  async getModels(projectId?: string): Promise<ClearMLModel[]> {
    await this.ensureInitialized();
    
    try {
      let url = `${this.apiUrl}/models`;
      if (projectId) {
        url += `?project_id=${projectId}`;
      }
      
      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': this.authHeader
        }
      });
      
      if (!response.ok) {
        throw new Error(`Failed to get models: ${response.statusText}`);
      }
      
      const data = await response.json();
      return data.data?.models || [];
    } catch (error) {
      console.error('Error fetching ClearML models:', error);
      return [];
    }
  }

  /**
   * Get a specific model by ID
   */
  async getModel(modelId: string): Promise<ClearMLModel | null> {
    await this.ensureInitialized();
    
    try {
      const response = await fetch(`${this.apiUrl}/models/${modelId}`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': this.authHeader
        }
      });
      
      if (!response.ok) {
        throw new Error(`Failed to get model: ${response.statusText}`);
      }
      
      const data = await response.json();
      return data.data || null;
    } catch (error) {
      console.error(`Error fetching ClearML model ${modelId}:`, error);
      return null;
    }
  }

  /**
   * Get pipelines from ClearML
   */
  async getPipelines(projectId?: string): Promise<ClearMLPipeline[]> {
    await this.ensureInitialized();
    
    try {
      let url = `${this.apiUrl}/pipelines`;
      if (projectId) {
        url += `?project_id=${projectId}`;
      }
      
      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': this.authHeader
        }
      });
      
      if (!response.ok) {
        throw new Error(`Failed to get pipelines: ${response.statusText}`);
      }
      
      const data = await response.json();
      return data.data?.pipelines || [];
    } catch (error) {
      console.error('Error fetching ClearML pipelines:', error);
      return [];
    }
  }

  /**
   * Get a specific pipeline by ID
   */
  async getPipeline(pipelineId: string): Promise<ClearMLPipeline | null> {
    await this.ensureInitialized();
    
    try {
      const response = await fetch(`${this.apiUrl}/pipelines/${pipelineId}`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': this.authHeader
        }
      });
      
      if (!response.ok) {
        throw new Error(`Failed to get pipeline: ${response.statusText}`);
      }
      
      const data = await response.json();
      return data.data || null;
    } catch (error) {
      console.error(`Error fetching ClearML pipeline ${pipelineId}:`, error);
      return null;
    }
  }

  /**
   * Get the latest completed task for a given project and task name pattern
   */
  async getLatestTask(projectId: string, namePattern: string): Promise<ClearMLTask | null> {
    await this.ensureInitialized();
    
    try {
      const tasks = await this.getTasks(projectId, ['completed']);
      
      // Filter tasks by name pattern and sort by last_update (descending)
      const filteredTasks = tasks
        .filter(task => task.name.includes(namePattern))
        .sort((a, b) => new Date(b.last_update).getTime() - new Date(a.last_update).getTime());
      
      return filteredTasks.length > 0 ? filteredTasks[0] : null;
    } catch (error) {
      console.error(`Error finding latest task in project ${projectId}:`, error);
      return null;
    }
  }

  /**
   * Get the Web UI URL for a task
   */
  getTaskWebUrl(taskId: string): string {
    return `${this.webUrl}/projects/task/${taskId}`;
  }

  /**
   * Get the Web UI URL for a model
   */
  getModelWebUrl(modelId: string): string {
    return `${this.webUrl}/models/${modelId}`;
  }

  /**
   * Get the Web UI URL for a project
   */
  getProjectWebUrl(projectId: string): string {
    return `${this.webUrl}/projects/${projectId}`;
  }

  /**
   * Get the best model based on a specific metric
   * Similar to MLflow's registered models concept
   */
  async getBestModel(projectId: string, metricName: string, optimizeFor: 'min' | 'max' = 'min'): Promise<any> {
    await this.ensureInitialized();
    
    try {
      // Get all completed tasks for the project
      const tasks = await this.getTasks(projectId, ['completed']);
      
      if (tasks.length === 0) {
        return null;
      }
      
      // Get metrics for each task
      const tasksWithMetrics = await Promise.all(
        tasks.map(async (task) => {
          const metrics = await this.getTaskMetrics(task.id);
          return { ...task, metrics };
        })
      );
      
      // Filter tasks that have the

