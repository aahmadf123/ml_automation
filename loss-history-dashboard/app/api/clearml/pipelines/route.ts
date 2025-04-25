import { NextResponse } from 'next/server'
import { ClearMLService } from '@/lib/services/clearml'

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const projectId = searchParams.get('project_id');
    const limit = parseInt(searchParams.get('limit') || '20', 10);
    
    const service = new ClearMLService();
    await service.initialize();
    
    // If no project ID is provided, get all pipelines
    if (!projectId) {
      const pipelines = await service.getPipelines();
      // Apply limit
      const limitedPipelines = pipelines.slice(0, limit);
      
      return NextResponse.json({
        pipelines: limitedPipelines,
        pipeline_count: limitedPipelines.length,
        total_count: pipelines.length
      });
    }
    
    // If project ID is provided, get pipelines for that project
    const pipelines = await service.getPipelines(projectId);
    
    // Apply limit
    const limitedPipelines = pipelines.slice(0, limit);
    
    // Get project details
    const projects = await service.getProjects();
    const project = projects.find(p => p.id === projectId);
    
    return NextResponse.json({
      project_id: projectId,
      project_name: project?.name || 'Unknown Project',
      pipelines: limitedPipelines,
      pipeline_count: limitedPipelines.length,
      total_count: pipelines.length,
      web_url: service.getProjectWebUrl(projectId)
    });
  } catch (error) {
    console.error('Error fetching pipelines from ClearML:', error);
    return NextResponse.json(
      { error: 'Failed to fetch pipelines from ClearML' },
      { status: 500 }
    );
  }
}

// GET endpoint for a specific pipeline
export async function GET_PIPELINE(request: Request, { params }: { params: { id: string } }) {
  try {
    const pipelineId = params.id;
    
    const service = new ClearMLService();
    await service.initialize();
    
    const pipeline = await service.getPipeline(pipelineId);
    if (!pipeline) {
      return NextResponse.json(
        { error: 'Pipeline not found' },
        { status: 404 }
      );
    }
    
    return NextResponse.json({
      ...pipeline,
      web_url: `${service.webUrl}/pipelines/${pipelineId}`
    });
  } catch (error) {
    console.error(`Error fetching pipeline ${params.id} from ClearML:`, error);
    return NextResponse.json(
      { error: 'Failed to fetch pipeline from ClearML' },
      { status: 500 }
    );
  }
}

