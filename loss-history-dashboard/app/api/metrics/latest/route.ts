import { NextResponse } from 'next/server'
import { ClearMLService } from '@/lib/services/clearml'

export async function GET() {
  try {
    const service = new ClearMLService();
    await service.initialize();
    
    // Get projects (equivalent to MLflow experiments)
    const projects = await service.getProjects();
    if (!projects || projects.length === 0) {
      return NextResponse.json(
        { error: 'No projects found' },
        { status: 404 }
      );
    }
    
    // Use first project by default (can be parameterized later)
    const projectId = projects[0].id;
    
    // Get the latest completed task from this project
    // We'll filter for training tasks (adapt the pattern as needed)
    const latestTask = await service.getLatestTask(projectId, 'train');
    
    if (!latestTask) {
      return NextResponse.json(
        { error: 'No completed tasks found for this project' },
        { status: 404 }
      );
    }
    
    // Get metrics for this task
    const metrics = await service.getTaskMetrics(latestTask.id);
    
    return NextResponse.json({
      run_id: latestTask.id,
      start_time: latestTask.started,
      end_time: latestTask.completed,
      status: service.mapTaskStatus(latestTask.status),
      metrics: metrics,
      params: latestTask.hyperparams || {},
      task_name: latestTask.name,
      project_id: projectId,
      project_name: latestTask.project_name,
      web_url: service.getTaskWebUrl(latestTask.id)
    });
  } catch (error) {
    console.error('Error fetching metrics from ClearML:', error);
    return NextResponse.json(
      { error: 'Failed to fetch metrics from ClearML' },
      { status: 500 }
    );
  }
}
