import { MLflowClient } from '@mlflow/client'

const mlflowClient = new MLflowClient({
  baseUrl: process.env.NEXT_PUBLIC_MLFLOW_TRACKING_URI || 'http://localhost:5000',
})

export interface ModelMetrics {
  rmse: { value: number; change: number; threshold: number; warning: boolean }
  mse: { value: number; change: number; threshold: number; warning: boolean }
  mae: { value: number; change: number; threshold: number; warning: boolean }
  r2: { value: number; change: number; threshold: number; warning: boolean }
}

export interface ModelData {
  [modelId: string]: ModelMetrics
}

export async function fetchModelMetrics(): Promise<ModelData> {
  try {
    // Get all experiments
    const experiments = await mlflowClient.searchExperiments()
    
    // Get the latest run for each model
    const modelData: ModelData = {}
    
    for (const experiment of experiments) {
      if (!experiment.name.startsWith('model')) continue
      
      const runs = await mlflowClient.searchRuns({
        experimentIds: [experiment.experiment_id],
        orderBy: ['start_time DESC'],
        limit: 2 // Get latest run and previous run for comparison
      })
      
      if (runs.length === 0) continue
      
      const latestRun = runs[0]
      const previousRun = runs[1]
      
      // Extract metrics
      const metrics = {
        rmse: {
          value: parseFloat(latestRun.data.metrics['rmse']),
          change: previousRun ? 
            parseFloat(latestRun.data.metrics['rmse']) - parseFloat(previousRun.data.metrics['rmse']) : 0,
          threshold: 0.5,
          warning: parseFloat(latestRun.data.metrics['rmse']) > 0.5
        },
        mse: {
          value: parseFloat(latestRun.data.metrics['mse']),
          change: previousRun ? 
            parseFloat(latestRun.data.metrics['mse']) - parseFloat(previousRun.data.metrics['mse']) : 0,
          threshold: 0.25,
          warning: parseFloat(latestRun.data.metrics['mse']) > 0.25
        },
        mae: {
          value: parseFloat(latestRun.data.metrics['mae']),
          change: previousRun ? 
            parseFloat(latestRun.data.metrics['mae']) - parseFloat(previousRun.data.metrics['mae']) : 0,
          threshold: 0.4,
          warning: parseFloat(latestRun.data.metrics['mae']) > 0.4
        },
        r2: {
          value: parseFloat(latestRun.data.metrics['r2']),
          change: previousRun ? 
            parseFloat(latestRun.data.metrics['r2']) - parseFloat(previousRun.data.metrics['r2']) : 0,
          threshold: 0.8,
          warning: parseFloat(latestRun.data.metrics['r2']) < 0.8
        }
      }
      
      modelData[experiment.name] = metrics
    }
    
    return modelData
  } catch (error) {
    console.error('Error fetching model metrics:', error)
    throw error
  }
}

export async function fetchModelHistory(modelId: string, metric: string): Promise<{ timestamp: number; value: number }[]> {
  try {
    const experiment = await mlflowClient.searchExperiments({
      filter: `name = '${modelId}'`
    })
    
    if (experiment.length === 0) {
      throw new Error(`Experiment ${modelId} not found`)
    }
    
    const runs = await mlflowClient.searchRuns({
      experimentIds: [experiment[0].experiment_id],
      orderBy: ['start_time ASC']
    })
    
    return runs.map(run => ({
      timestamp: new Date(run.info.start_time).getTime(),
      value: parseFloat(run.data.metrics[metric])
    }))
  } catch (error) {
    console.error('Error fetching model history:', error)
    throw error
  }
}

export async function retrainModel(modelId: string): Promise<void> {
  try {
    const response = await fetch('/api/retrain', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ modelId })
    })
    
    if (!response.ok) {
      throw new Error('Failed to initiate retraining')
    }
  } catch (error) {
    console.error('Error initiating retraining:', error)
    throw error
  }
} 