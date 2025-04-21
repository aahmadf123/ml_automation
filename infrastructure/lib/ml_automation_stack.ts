import * as cdk from 'aws-cdk-lib';
import * as mwaa from 'aws-cdk-lib/aws-mwaa';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as apigatewayv2 from 'aws-cdk-lib/aws-apigatewayv2';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import * as sagemaker from 'aws-cdk-lib/aws-sagemaker';
import * as cloudwatch from 'aws-cdk-lib/aws-cloudwatch';
import * as sns from 'aws-cdk-lib/aws-sns';
import * as subscriptions from 'aws-cdk-lib/aws-sns-subscriptions';
import * as apigatewayv2_integrations from 'aws-cdk-lib/aws-apigatewayv2-integrations';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import { Construct } from 'constructs';
import { SubscriptionProtocol } from 'aws-cdk-lib/aws-sns';

export class MlAutomationStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // Create VPC for MWAA
    const vpc = new ec2.Vpc(this, 'MwaaVpc', {
      maxAzs: 2,
      natGateways: 1,
    });

    // Create security group for MWAA
    const securityGroup = new ec2.SecurityGroup(this, 'MwaaSecurityGroup', {
      vpc,
      description: 'Security group for MWAA environment',
      allowAllOutbound: true,
    });

    // Create S3 bucket for DAGs
    const dagsBucket = new s3.Bucket(this, 'DagsBucket', {
      versioned: true,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      encryption: s3.BucketEncryption.S3_MANAGED,
    });

    // Create S3 bucket for model artifacts
    const modelBucket = new s3.Bucket(this, 'ModelBucket', {
      versioned: true,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      encryption: s3.BucketEncryption.S3_MANAGED,
    });

    // Import existing secrets for Airflow
    const airflowSecrets = secretsmanager.Secret.fromSecretNameV2(
      this, 'AirflowSecrets', 'airflow-secrets'
    );

    // Create secrets for dashboard
    const dashboardSecrets = new secretsmanager.Secret(this, 'DashboardSecrets', {
      secretName: 'dashboard-secrets',
      description: 'Secrets for dashboard',
      generateSecretString: {
        secretStringTemplate: JSON.stringify({
          NEXT_PUBLIC_WEBSOCKET_URL: '',
          NEXT_PUBLIC_API_URL: '',
        }),
        generateStringKey: 'password',
      },
    });

    // Create MWAA environment
    const mwaaEnvironment = new mwaa.CfnEnvironment(this, 'MwaaEnvironment', {
      name: 'ml-automation-environment',
      airflowVersion: '2.10.5',
      sourceBucketArn: dagsBucket.bucketArn,
      dagS3Path: 'dags',  // This is where your DAG files will be stored in S3
      executionRoleArn: this.createMwaaRole().roleArn,
      webserverAccessMode: 'PUBLIC_ONLY',
      networkConfiguration: {
        subnetIds: vpc.privateSubnets.map(subnet => subnet.subnetId),
        securityGroupIds: [securityGroup.securityGroupId],
      },
      loggingConfiguration: {
        dagProcessingLogs: {
          enabled: true,
          logLevel: 'INFO',
        },
        schedulerLogs: {
          enabled: true,
          logLevel: 'INFO',
        },
        taskLogs: {
          enabled: true,
          logLevel: 'INFO',
        },
        webserverLogs: {
          enabled: true,
          logLevel: 'INFO',
        },
        workerLogs: {
          enabled: true,
          logLevel: 'INFO',
        },
      },
    });

    // Create Lambda functions
    const connectFunction = new lambda.Function(this, 'ConnectFunction', {
      runtime: lambda.Runtime.PYTHON_3_9,
      handler: 'index.handler',
      code: lambda.Code.fromAsset('lambda/connect'),
      environment: {
        LOG_GROUP: '/ml-automation/websocket-connections'
      }
    });

    const disconnectFunction = new lambda.Function(this, 'DisconnectFunction', {
      runtime: lambda.Runtime.PYTHON_3_9,
      handler: 'index.handler',
      code: lambda.Code.fromAsset('lambda/disconnect'),
      environment: {
        LOG_GROUP: '/ml-automation/websocket-connections'
      }
    });

    const driftEventFunction = new lambda.Function(this, 'DriftEventFunction', {
      runtime: lambda.Runtime.PYTHON_3_9,
      handler: 'index.handler',
      code: lambda.Code.fromAsset('lambda/drift-event'),
      environment: {
        LOG_GROUP: '/ml-automation/websocket-connections',
        DRIFT_LOG_GROUP: '/ml-automation/drift-events'
      }
    });

    const defaultFunction = new lambda.Function(this, 'DefaultFunction', {
      runtime: lambda.Runtime.PYTHON_3_9,
      handler: 'index.handler',
      code: lambda.Code.fromAsset('lambda/default'),
      environment: {
        LOG_GROUP: '/ml-automation/websocket-connections'
      }
    });

    // Create WebSocket API
    const webSocketApi = new apigatewayv2.WebSocketApi(this, 'MLAutomationWebSocketApi', {
      apiName: 'ML Automation WebSocket API',
      connectRouteOptions: {
        integration: new apigatewayv2_integrations.WebSocketLambdaIntegration('ConnectIntegration', connectFunction)
      },
      disconnectRouteOptions: {
        integration: new apigatewayv2_integrations.WebSocketLambdaIntegration('DisconnectIntegration', disconnectFunction)
      },
      defaultRouteOptions: {
        integration: new apigatewayv2_integrations.WebSocketLambdaIntegration('DefaultIntegration', defaultFunction)
      }
    });

    // Add custom route for drift events
    webSocketApi.addRoute('driftEvent', {
      integration: new apigatewayv2_integrations.WebSocketLambdaIntegration('DriftEventIntegration', driftEventFunction)
    });

    // Create WebSocket Stage
    const stage = new apigatewayv2.WebSocketStage(this, 'WebSocketStage', {
      webSocketApi: webSocketApi,
      stageName: 'prod',
      autoDeploy: true
    });

    // Grant permissions to Lambda functions
    connectFunction.addToRolePolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'logs:CreateLogGroup',
        'logs:CreateLogStream',
        'logs:PutLogEvents'
      ],
      resources: ['*']
    }));

    disconnectFunction.addToRolePolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'logs:CreateLogGroup',
        'logs:CreateLogStream',
        'logs:PutLogEvents'
      ],
      resources: ['*']
    }));

    driftEventFunction.addToRolePolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'logs:CreateLogGroup',
        'logs:CreateLogStream',
        'logs:PutLogEvents',
        'cloudwatch:PutMetricData'
      ],
      resources: ['*']
    }));

    // Create SNS topic for alerts
    const alertTopic = new sns.Topic(this, 'AlertTopic', {
      displayName: 'ML Automation Alerts',
    });

    // Add Slack subscription using the secret value
    alertTopic.addSubscription(
      new subscriptions.UrlSubscription(
        cdk.SecretValue.secretsManager(airflowSecrets.secretName, {
          jsonField: 'SLACK_WEBHOOK_URL'
        }).toString(),
        { protocol: sns.SubscriptionProtocol.HTTPS }
      )
    );

    // Create CloudWatch dashboard
    const dashboard = new cloudwatch.Dashboard(this, 'MLAutomationDashboard', {
      dashboardName: 'ML-Automation-Dashboard',
    });

    // Add metrics to dashboard
    dashboard.addWidgets(
      new cloudwatch.GraphWidget({
        title: 'Model Performance',
        left: [
          new cloudwatch.Metric({
            namespace: 'MLAutomation',
            metricName: 'RMSE',
            statistic: 'Average',
            period: cdk.Duration.minutes(5),
          }),
        ],
      }),
      new cloudwatch.GraphWidget({
        title: 'Data Drift',
        left: [
          new cloudwatch.Metric({
            namespace: 'MLAutomation',
            metricName: 'DriftRate',
            statistic: 'Average',
            period: cdk.Duration.minutes(5),
          }),
        ],
      })
    );

    // Output important values
    new cdk.CfnOutput(this, 'MwaaWebserverUrl', {
      value: `https://${mwaaEnvironment.attrWebserverUrl}`,
      description: 'MWAA Web Server URL',
    });

    new cdk.CfnOutput(this, 'WebSocketApiUrl', {
      value: stage.url,
      description: 'WebSocket API URL'
    });
  }

  private createMwaaRole(): iam.Role {
    const role = new iam.Role(this, 'MwaaExecutionRole', {
      assumedBy: new iam.ServicePrincipal('airflow-env.amazonaws.com'),
    });

    // Add permissions for S3
    role.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          's3:ListBucket',
          's3:GetObject',
          's3:PutObject',
          's3:DeleteObject',
        ],
        resources: ['arn:aws:s3:::ml-automation-*/*'],
      })
    );

    // Add permissions for CloudWatch
    role.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          'cloudwatch:PutMetricData',
          'logs:CreateLogStream',
          'logs:CreateLogGroup',
          'logs:PutLogEvents',
          'logs:GetLogEvents',
          'logs:GetLogRecord',
          'logs:GetLogGroupFields',
          'logs:GetQueryResults',
        ],
        resources: ['*'],
      })
    );

    // Add permissions for Secrets Manager
    role.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: ['secretsmanager:GetSecretValue'],
        resources: ['arn:aws:secretsmanager:*:*:secret:airflow-secrets-*'],
      })
    );

    return role;
  }
} 