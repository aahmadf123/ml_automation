import { NextResponse } from 'next/server'
import { IntegrationsService } from '@/lib/services/integrations'

export async function POST(request: Request) {
  try {
    const { 
      taskId, 
      runId, 
      approved,
      approver,
      comment 
    } = await request.json();

    if (!taskId || !runId || approved === undefined || !approver) {
      return NextResponse.json(
        { error: 'Task ID, Run ID, approval decision, and approver are required' },
        { status: 400 }
      );
    }

    const service = new IntegrationsService();
    await service.initialize();
    
    // Record the approval in Airflow
    const approvalData = {
      task_id: taskId,
      run_id: runId,
      approved: approved,
      approver: approver,
      comment: comment || ''
    };
    
    // Trigger variables update in Airflow to record the approval
    const status = approved ? "approved" : "rejected";
    const variablePrefix = "HITL_ACTION_";
    const variableKey = `${variablePrefix}${taskId}_${runId}`;
    
    // Update the Airflow variable
    await service.setAirflowVariable(variableKey, status);
    
    if (comment) {
      const commentKey = `HITL_COMMENT_${taskId}_${runId}`;
      await service.setAirflowVariable(commentKey, comment);
    }
    
    // Log the approval to ClearML
    const clearmlTaskId = await service.logApprovalToClearML({
      taskId,
      runId,
      approved,
      approver,
      comment: comment || '',
      timestamp: new Date().toISOString()
    });
    
    // Also log a Slack notification if configured
    await service.sendSlackMessage({
      channel: 'ml-approvals',
      text: `${approved ? '✅ Approved' : '❌ Rejected'}: Task ${taskId} in run ${runId} by ${approver}${comment ? `. Comment: ${comment}` : ''}`
    });

    return NextResponse.json({
      success: true,
      message: `${approved ? 'Approval' : 'Rejection'} recorded successfully`,
      clearmlTaskId,
      taskId,
      runId
    });
  } catch (error) {
    console.error('Error recording approval decision:', error);
    return NextResponse.json(
      { error: 'Failed to record approval decision' },
      { status: 500 }
    );
  }
} 