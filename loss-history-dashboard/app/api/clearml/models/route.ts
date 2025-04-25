import { NextResponse } from 'next/server'
import { ClearMLService } from '@/lib/services/clearml'

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const projectId = searchParams.get('project_id');
    const limit = parseInt(searchParams.get('limit') || '20', 10);
    const metricName = searchParams.get('metric') || 'rmse';
    const optimizeFor = (searchParams.get('optimize') || 'min') as 'min' | 'max';
    
    const service = new ClearMLService();
    await service.initialize();
    
    // If the 'best' parameter is set, return the best model based on the metric
    if (searchParams.get('best') === 'true' && projectId) {
      const bestModel = await service.getBestModel(projectId, metricName, optimizeFor);
      
      if (!bestModel) {
        return NextResponse.json(
          { error: 'No models found with the specified metric' },
          { status: 404 }
        );
      }
      
      return NextResponse.json({
        model_id: bestModel.id,
        task_id: bestModel.task_id,
        name: bestModel.name,
        framework: bestModel.framework,
        metrics: bestModel.metrics,
        uri: bestModel.uri,
        web_url: service.getModelWebUrl(bestModel.id)
      });
    }
    
    // If no project ID is provided, get all models
    if (!projectId) {
      const models = await service.getModels();
      
      // Apply limit
      const limitedModels = models.slice(0, limit);
      
      return NextResponse.json({
        models: limitedModels,
        model_count: limitedModels.length,
        total_count: models.length
      });
    }
    
    // If project ID is provided, get models for that project
    const models = await service.getModels(projectId);
    
    // Apply limit
    const limitedModels = models.slice(0, limit);
    
    // Get project details
    const projects = await service.getProjects();
    const project = projects.find(p => p.id === projectId);
    
    return NextResponse.json({
      project_id: projectId,
      project_name: project?.name || 'Unknown Project',
      models: limitedModels,
      model_count: limitedModels.length,
      total_count: models.length,
      web_url: service.getProjectWebUrl(projectId)
    });
  } catch (error) {
    console.error('Error fetching models from ClearML:', error);
    return NextResponse.json(
      { error: 'Failed to fetch models from ClearML' },
      { status: 500 }
    );
  }
}

// GET endpoint for a specific model
export async function GET_MODEL(request: Request, { params }: { params: { id: string } }) {
  try {
    const modelId = params.id;
    
    const service = new ClearMLService();
    await service.initialize();
    
    const model = await service.getModel(modelId);
    if (!model) {
      return NextResponse.json(
        { error: 'Model not found' },
        { status: 404 }
      );
    }
    
    // If this model is linked to a task, get the task metrics too
    let metrics = {};
    if (model.task_id) {
      metrics = await service.getTaskMetrics(model.task_id);
    }
    
    return NextResponse.json({
      ...model,
      metrics,
      web_url: service.getModelWebUrl(modelId)
    });
  } catch (error) {
    console.error(`Error fetching model ${params.id} from ClearML:`, error);
    return NextResponse.json(
      { error: 'Failed to fetch model from ClearML' },
      { status: 500 }
    );
  }
}

