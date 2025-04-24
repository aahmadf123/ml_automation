import { NextResponse } from 'next/server'
import { AirflowClient } from '@/lib/airflow'

export async function POST(request: Request) {
  try {
    const { modelId } = await request.json()

    if (!modelId) {
      return NextResponse.json(
        { error: 'Model ID is required' },
        { status: 400 }
      )
    }

    // Get a client from secrets
    const airflowClient = await AirflowClient.fromSecrets()

    // Trigger the retraining DAG
    const response = await airflowClient.triggerDag({
      dagId: 'homeowner_loss_history_full_pipeline',
      conf: {
        model_id: modelId,
        action: 'retrain'
      }
    })

    return NextResponse.json({
      message: 'Retraining initiated',
      dagRunId: response.dagRunId
    })
  } catch (error) {
    console.error('Error initiating retraining:', error)
    return NextResponse.json(
      { error: 'Failed to initiate retraining' },
      { status: 500 }
    )
  }
} 