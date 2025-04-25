    const metric = searchParams.get('metric') || 'rmse';
    const optimizeFor = searchParams.get('optimize') || 'min'; // 'min' or 'max'
    
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
    
    // Use the primary experiment
    const experimentId = experiments[0].experiment_id;
    
    // Get all model runs
    const runs = await service.getRuns(experimentId);
    if (!runs || runs.length === 0) {
      return NextResponse.json(
        { error: 'No runs found for this experiment' },
        { status: 404 }
      );
    }
    
    // Filter for completed runs with the specified metric
    const validRuns = runs.filter(run => 
      run.status === 'FINISHED' && 
      run.metrics && 
      run.metrics[metric] !== undefined
    );
    
    if (validRuns.length === 0) {
      return NextResponse.json(
        { error: `No runs found with metric: ${metric}` },
        { status: 404 }
      );
    }
    
    // Sort runs by the metric (either ascending or descending based on optimization goal)
    const sortedRuns = validRuns.sort((a, b) => {
      if (optimizeFor === 'min') {
        return a.metrics[metric] - b.metrics[metric];
      } else {
        return b.metrics[metric] - a.metrics[metric];
      }
    });
    
    // Get the best run
    const bestRun = sortedRuns[0];
    
    // Check if this run has a registered model
    const modelInfo = bestRun.params && bestRun.params.model_id 
      ? { model_id: bestRun.params.model_id } 
      : { model_id: `model_${bestRun.run_id}` };
    
    return NextResponse.json({
      run_id: bestRun.run_id,
      ...modelInfo,
      status: bestRun.status,
      metrics: bestRun.metrics,
      optimized_for: {
        metric: metric,
        goal: optimizeFor,
        value: bestRun.metrics[metric]
      },
      start_time: bestRun.start_time,
      end_time: bestRun.end_time
    });
  } catch (error) {
    console.error('Error fetching current best model:', error);
    return NextResponse.json(
      { error: 'Failed to fetch current model' },
      { status: 500 }
    );
  }
} 