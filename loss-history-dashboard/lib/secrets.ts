import { SecretsManager } from 'aws-sdk';

let cachedSecrets: Record<string, any> | null = null;

export async function getSecrets(): Promise<Record<string, any>> {
  if (cachedSecrets) {
    return cachedSecrets;
  }

  // Use environment variable for local development fallback
  if (process.env.NODE_ENV === 'development' && process.env.USE_LOCAL_ENV === 'true') {
    console.log('Using local environment variables for development');
    return {
      AIRFLOW_API_URL: process.env.AIRFLOW_API_URL,
      AIRFLOW_USERNAME: process.env.AIRFLOW_USERNAME,
      AIRFLOW_PASSWORD: process.env.AIRFLOW_PASSWORD,
      MLFLOW_API_URL: process.env.MLFLOW_API_URL,
      SLACK_TOKEN: process.env.SLACK_TOKEN,
      CLEARML_API_URL: process.env.CLEARML_API_URL,
      CLEARML_ACCESS_KEY: process.env.CLEARML_ACCESS_KEY,
      CLEARML_SECRET_KEY: process.env.CLEARML_SECRET_KEY,
      WEBSOCKET_ENDPOINT: process.env.WEBSOCKET_ENDPOINT
    };
  }

  try {
    const region = process.env.AWS_REGION || 'us-east-1';
    const secretName = 'dashboard-secrets';
    
    const client = new SecretsManager({
      region: region
    });
    
    const response = await client.getSecretValue({ SecretId: secretName }).promise();
    
    if ('SecretString' in response) {
      cachedSecrets = JSON.parse(response.SecretString);
      return cachedSecrets;
    } else {
      throw new Error('Secret value is not a string');
    }
  } catch (error) {
    console.error('Error retrieving secrets:', error);
    throw error;
  }
} 