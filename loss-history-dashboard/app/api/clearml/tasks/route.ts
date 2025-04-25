import { NextResponse } from 'next/server'
import { ClearMLService } from '@/lib/services/clearml'

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const projectId = searchParams.get('project_id');
    const status = searchParams.get('status');
    const limit = parseInt(searchParams.get('limit') || '20', 10);
    const namePattern = searchParams.get('name_pattern');
    
    const service = new ClearMLService();
    await service.initialize();
    
    // If no project ID is provided, get all projects and return the first set of tasks
    if (!projectId) {
      const projects = await service.getProjects();
      if (!projects || projects.length === 0) {
        return NextResponse.json(
          { error: 'No projects found' },
          { status: 404 }
        );
      }
      
      // Use the first project by default
      const firstProjectId = projects[0].id;
      
      // Get tasks for this project
      const statusFilter = status ? status.split(',') : undefined;
      let tasks = await service.getTasks(firstProjectId, statusFilter);
      
      // Apply name pattern filter if provided
      if (namePattern && tasks.length > 0) {
        tasks = tasks.filter(task => task.name.includes(namePattern));
      }
      
      // Apply limit
      tasks = tasks.slice(0, limit);
      
      return NextResponse.json({
        project_id: firstProjectId,
        project_name: projects[0].name,
        tasks: tasks,
        task_count: tasks.length,
        total_count: tasks.length,
        web_url: service.getProjectWebUrl(firstProjectId)
      });
    }
    
    // If project ID is provided, get tasks for that project
    const statusFilter = status ? status.split(',') : undefined;
    let tasks = await service.getTasks(projectId, statusFilter);
    
    // Apply name pattern filter if provided
    if (namePattern && tasks.length > 0) {
      tasks = tasks.filter(task => task.name.includes(namePattern));
    }
    
    // Apply limit
    tasks = tasks.slice(0, limit);
    
    // Get project details
    const projects = await service.getProjects();
    const project = projects.find(p => p.id === projectId);
    
    return NextResponse.json({
      project_id: projectId,
      project_name: project?.name || 'Unknown Project',
      tasks: tasks,
      task_count: tasks.length,
      total_count: tasks.length,
      web_url: service.getProjectWebUrl(projectId)
    });
  } catch (error) {
    console.error('Error fetching tasks from ClearML:', error);
    return NextResponse.json(
      { error: 'Failed to fetch tasks from ClearML' },
      { status: 500 }
    );
  }
}

// GET endpoint for a specific task
export async function GET_TASK(request: Request, { params }: { params: { id: string } }) {
  try {
    const taskId = params.id;
    
    const service = new ClearMLService();
    await service.initialize();
    
    const task = await service.getTask(taskId);
    if (!task) {
      return NextResponse.json(
        { error: 'Task not found' },
        { status: 404 }
      );
    }
    
    // Get metrics for this task
    const metrics = await service.getTaskMetrics(taskId);
    
    return NextResponse.json({
      ...task,
      metrics: metrics,
      web_url: service.getTaskWebUrl(taskId)
    });
  } catch (error) {
    console.error(`Error fetching task ${params.id} from ClearML:`, error);
    return NextResponse.json(
      { error: 'Failed to fetch task from ClearML' },
      { status: 500 }
    );
  }
}

