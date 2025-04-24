import { NextResponse } from 'next/server'
import { IntegrationsService } from '@/lib/services/integrations'

export async function POST(request: Request) {
  try {
    const { 
      modelId, 
      overrideReason, 
      userId,
      approvedBy,
      metrics,
      notes 
    } = await request.json();

    if (!modelId || !overrideReason) {
      return NextResponse.json(
        { error: 'Model ID and override reason are required' },
        { status: 400 }
      );
    }

    const service = new IntegrationsService();
    await service.initialize();
    
    // Log the override to ClearML
    const clearmlTaskId = await service.logOverrideToClearML({
      modelId,
      overrideReason,
      userId: userId || 'unknown',
      approvedBy: approvedBy || null,
      metrics: metrics || {},
      notes: notes || '',
      timestamp: new Date().toISOString()
    });
    
    // Also log a Slack notification if configured
    await service.sendSlackMessage({
      channel: 'model-approvals',
      text: `Model Override: ${modelId} was manually overridden by ${userId || 'a user'}. Reason: ${overrideReason}`
    });

    // Trigger an Airflow DAG to handle the override if needed
    await service.triggerDAG('model_override_handler', {
      model_id: modelId,
      override_reason: overrideReason,
      user_id: userId,
      override_time: new Date().toISOString()
    });

    return NextResponse.json({
      success: true,
      message: 'Manual override recorded successfully',
      clearmlTaskId,
      modelId
    });
  } catch (error) {
    console.error('Error recording manual override:', error);
    return NextResponse.json(
      { error: 'Failed to record manual override' },
      { status: 500 }
    );
  }
} 