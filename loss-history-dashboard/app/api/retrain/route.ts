import { NextResponse } from 'next/server'
import { AirflowClient } from '@/lib/airflow'

const airflowClient = new AirflowClient({
  baseUrl: process.env.AIRFLOW_API_URL || 'http://localhost:8080',
  username: process.env.AIRFLOW_USERNAME || 'airflow',
  password: process.env.AIRFLOW_PASSWORD || 'airflow'
})

export async function POST(request: Request) {
  try {
    const { modelId } = await request.json()

    if (!modelId) {
      return NextResponse.json(
        { error: 'Model ID is required' },
        { status: 400 }
      )
    }

    // Determine the environment (EC2 or localhost)
    const isEC2 = process.env.EC2_INSTANCE === 'true'

    // Trigger the retraining DAG with appropriate configuration
    const response = await airflowClient.triggerDag({
      dagId: 'homeowner_loss_history_full_pipeline',
      conf: {
        model_id: modelId,
        action: 'retrain',
        environment: isEC2 ? 'ec2' : 'localhost'
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
