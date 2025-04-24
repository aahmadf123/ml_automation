import { NextResponse } from 'next/server'
import { IntegrationsService } from '@/lib/services/integrations'

export async function GET() {
  try {
    const service = new IntegrationsService();
    await service.initialize();
    
    // Get experiments first
    const experiments = await service.getExperiments();
    if (!experiments || experiments.length === 0) {
      return NextResponse.json(
        { error: 'No experiments found' },
        { status: 404 }
      );
    }
    
    // Use first experiment by default (can be parameterized later)
    const experimentId = experiments[0].experiment_id;
    
    // Get the latest runs
    const runs = await service.getRuns(experimentId);
    if (!runs || runs.length === 0) {
      return NextResponse.json(
        { error: 'No runs found for this experiment' },
        { status: 404 }
      );
    }
    
    // Sort runs by start_time (descending) to get the most recent
    const sortedRuns = runs.sort((a, b) => 
      new Date(b.start_time).getTime() - new Date(a.start_time).getTime()
    );
    
    // Get the latest run and its metrics
    const latestRun = sortedRuns[0];
    
    return NextResponse.json({
      run_id: latestRun.run_id,
      start_time: latestRun.start_time,
      end_time: latestRun.end_time,
      status: latestRun.status,
      metrics: latestRun.metrics,
      params: latestRun.params
    });
  } catch (error) {
    console.error('Error fetching metrics:', error);
    return NextResponse.json(
      { error: 'Failed to fetch metrics' },
      { status: 500 }
    );
  }
} 