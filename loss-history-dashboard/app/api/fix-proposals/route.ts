import { NextResponse } from 'next/server';
import { IntegrationsService } from '@/lib/services/integrations';

/**
 * GET handler for fetching fix proposals
 */
export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const status = searchParams.get('status') || 'pending';
    const modelId = searchParams.get('modelId');
    const limit = parseInt(searchParams.get('limit') || '10');
    
    const service = new IntegrationsService();
    const proposals = await service.getFixProposals(status, modelId, limit);
    
    return NextResponse.json({ 
      success: true, 
      data: proposals 
    });
    
  } catch (error) {
    console.error('Error fetching fix proposals:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to fetch fix proposals' }, 
      { status: 500 }
    );
  }
}

/**
 * POST handler for recording fix proposal decisions
 */
export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { 
      proposalId, 
      decision, 
      decidedBy, 
      comment 
    } = body;
    
    // Validate required fields
    if (!proposalId || !decision || !decidedBy) {
      return NextResponse.json(
        { success: false, error: 'Missing required fields' },
        { status: 400 }
      );
    }
    
    // Validate decision is either 'approve' or 'reject'
    if (decision !== 'approve' && decision !== 'reject') {
      return NextResponse.json(
        { success: false, error: "Decision must be either 'approve' or 'reject'" },
        { status: 400 }
      );
    }
    
    const service = new IntegrationsService();
    
    // Record the decision
    const result = await service.recordFixProposalDecision(
      proposalId,
      decision,
      decidedBy,
      comment
    );
    
    // If approved, implement the fix
    if (decision === 'approve') {
      // Get proposal details
      const proposal = await service.getFixProposal(proposalId);
      
      // Trigger fix implementation via Airflow
      await service.setAirflowVariable(
        `FIX_APPROVAL_${proposalId}`,
        JSON.stringify({
          proposalId,
          approvedBy: decidedBy,
          timestamp: new Date().toISOString(),
          comment: comment || ''
        })
      );
      
      // Send Slack notification about approved fix
      await service.sendSlackNotification(
        'ml-approvals',
        `*Fix Proposal Approved* 🛠️\n` +
        `ID: ${proposalId}\n` +
        `Problem: ${proposal?.problem || 'Unknown'}\n` +
        `Approved by: ${decidedBy}\n` +
        `Comment: ${comment || 'None'}`
      );
    } else {
      // Send Slack notification about rejected fix
      await service.sendSlackNotification(
        'ml-approvals',
        `*Fix Proposal Rejected* ❌\n` +
        `ID: ${proposalId}\n` +
        `Rejected by: ${decidedBy}\n` +
        `Comment: ${comment || 'None'}`
      );
    }
    
    return NextResponse.json({ 
      success: true, 
      data: result
    });
    
  } catch (error) {
    console.error('Error recording fix proposal decision:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to record fix proposal decision' }, 
      { status: 500 }
    );
  }
} 