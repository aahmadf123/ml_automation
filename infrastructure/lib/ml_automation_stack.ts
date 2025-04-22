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

    // Use existing S3 bucket for DAGs
    const dagsBucket = s3.Bucket.fromBucketName(this, 'ExistingDagsBucket', 'grange-seniordesign-bucket');

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
      airflowVersion: '2.10.1',
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
        'logs:PutLogEvents',
        'cloudwatch:PutMetricData',
        'execute-api:ManageConnections',
        'dynamodb:PutItem',
        'dynamodb:GetItem',
        'dynamodb:UpdateItem',
        'dynamodb:DeleteItem'
      ],
      resources: ['*']
    }));

    disconnectFunction.addToRolePolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'logs:CreateLogGroup',
        'logs:CreateLogStream',
        'logs:PutLogEvents',
        'cloudwatch:PutMetricData',
        'execute-api:ManageConnections',
        'dynamodb:PutItem',
        'dynamodb:GetItem',
        'dynamodb:UpdateItem',
        'dynamodb:DeleteItem'
      ],
      resources: ['*']
    }));

    driftEventFunction.addToRolePolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'logs:CreateLogGroup',
        'logs:CreateLogStream',
        'logs:PutLogEvents',
        'cloudwatch:PutMetricData',
        'execute-api:ManageConnections',
        's3:GetObject',
        's3:PutObject',
        's3:ListBucket',
        'sns:Publish',
        'dynamodb:PutItem',
        'dynamodb:GetItem',
        'dynamodb:UpdateItem',
        'dynamodb:DeleteItem'
      ],
      resources: [
        'arn:aws:s3:::grange-seniordesign-bucket',
        'arn:aws:s3:::grange-seniordesign-bucket/*',
        'arn:aws:sns:*:*:*'
      ]
    }));

    defaultFunction.addToRolePolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'logs:CreateLogGroup',
        'logs:CreateLogStream',
        'logs:PutLogEvents',
        'cloudwatch:PutMetricData',
        'execute-api:ManageConnections',
        'dynamodb:PutItem',
        'dynamodb:GetItem',
        'dynamodb:UpdateItem',
        'dynamodb:DeleteItem'
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

    // Create EC2 instance for Airflow and MLflow
    const ec2Instance = new ec2.Instance(this, 'AirflowMlflowInstance', {
      vpc,
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.MEDIUM),
      machineImage: ec2.MachineImage.latestAmazonLinux(),
      securityGroup,
      keyName: 'ec2-key-pair', // Replace with your key pair name
    });

    // Add user data to install and start Airflow and MLflow
    ec2Instance.addUserData(
      `#!/bin/bash
      sudo yum update -y
      sudo amazon-linux-extras install docker -y
      sudo service docker start
      sudo usermod -a -G docker ec2-user
      sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
      sudo chmod +x /usr/local/bin/docker-compose
      sudo yum install git -y
      git clone https://github.com/aahmadf123/ml_automation.git /home/ec2-user/ml_automation
      cd /home/ec2-user/ml_automation
      sudo docker-compose up -d
      `
    );

    // Output EC2 instance public DNS
    new cdk.CfnOutput(this, 'Ec2InstancePublicDns', {
      value: ec2Instance.instancePublicDnsName,
      description: 'EC2 Instance Public DNS',
    });
  }

  private createMwaaRole(): iam.Role {
    const role = new iam.Role(this, 'MwaaExecutionRole', {
      assumedBy: new iam.ServicePrincipal('airflow-env.amazonaws.com'),
    });

    // Add comprehensive S3 permissions
    role.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          // Basic S3 operations
          's3:ListBucket',
          's3:GetObject',
          's3:PutObject',
          's3:DeleteObject',
          's3:GetObjectVersion',
          's3:GetObjectTagging',
          's3:PutObjectTagging',
          's3:GetBucketLocation',
          's3:GetBucketVersioning',
          's3:ListAllMyBuckets',
          
          // Public access block permissions
          's3:GetAccountPublicAccessBlock',
          's3:GetBucketPublicAccessBlock',
          's3:PutBucketPublicAccessBlock',
          
          // Encryption permissions
          's3:GetEncryptionConfiguration',
          's3:PutEncryptionConfiguration',
          
          // Bucket policy permissions
          's3:GetBucketPolicy',
          's3:PutBucketPolicy',
          
          // CORS permissions
          's3:GetBucketCors',
          's3:PutBucketCors',
          
          // Lifecycle permissions
          's3:GetLifecycleConfiguration',
          's3:PutLifecycleConfiguration',
          
          // Logging permissions
          's3:GetBucketLogging',
          's3:PutBucketLogging',
          
          // Replication permissions
          's3:GetBucketReplication',
          's3:PutBucketReplication',
          
          // Tagging permissions
          's3:GetBucketTagging',
          's3:PutBucketTagging'
        ],
        resources: [
          'arn:aws:s3:::grange-seniordesign-bucket',
          'arn:aws:s3:::grange-seniordesign-bucket/*',
          'arn:aws:s3:::*',
          '*'
        ],
      })
    );

    // Add permissions for S3 Control (for account-level operations)
    role.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          's3:GetAccountPublicAccessBlock',
          's3:PutAccountPublicAccessBlock',
          's3:GetAccessPoint',
          's3:GetAccessPointPolicy',
          's3:GetAccessPointPolicyStatus',
          's3:ListAccessPoints'
        ],
        resources: ['*'],
      })
    );

    // Add specific permissions for the bucket
    role.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          's3:GetBucketPublicAccessBlock',
          's3:PutBucketPublicAccessBlock',
          's3:GetBucketPolicy',
          's3:PutBucketPolicy',
          's3:GetEncryptionConfiguration',
          's3:PutEncryptionConfiguration'
        ],
        resources: [
          'arn:aws:s3:::grange-seniordesign-bucket'
        ],
      })
    );

    // Add permissions for KMS (if using customer-managed keys)
    role.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          'kms:Decrypt',
          'kms:DescribeKey',
          'kms:Encrypt',
          'kms:GenerateDataKey',
          'kms:ReEncrypt',
          'kms:ReEncryptFrom',
          'kms:ReEncryptTo'
        ],
        resources: ['*'],
      })
    );

    // Add permissions for CloudWatch
    role.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          'cloudwatch:PutMetricData',
          'cloudwatch:GetMetricData',
          'cloudwatch:ListMetrics',
          'cloudwatch:GetMetricStatistics',
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

    // Add permissions for SNS
    role.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          'sns:Publish'
        ],
        resources: ['*'],
      })
    );

    // Add permissions for SageMaker
    role.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          'sagemaker:CreateModel',
          'sagemaker:CreateEndpoint',
          'sagemaker:CreateEndpointConfig',
          'sagemaker:DeleteEndpoint',
          'sagemaker:DeleteEndpointConfig',
          'sagemaker:DescribeEndpoint',
          'sagemaker:DescribeEndpointConfig',
          'sagemaker:DescribeModel',
          'sagemaker:InvokeEndpoint',
          'sagemaker:ListEndpoints',
          'sagemaker:ListEndpointConfigs',
          'sagemaker:ListModels',
          'sagemaker:UpdateEndpoint',
          'sagemaker:UpdateEndpointWeightsAndCapacities',
          'sagemaker:CreateTrainingJob',
          'sagemaker:DescribeTrainingJob',
          'sagemaker:StopTrainingJob',
          'sagemaker:ListTrainingJobs',
          'sagemaker:CreateHyperParameterTuningJob',
          'sagemaker:DescribeHyperParameterTuningJob',
          'sagemaker:StopHyperParameterTuningJob',
          'sagemaker:ListHyperParameterTuningJobs'
        ],
        resources: ['*'],
      })
    );

    // Add permissions for Step Functions (for workflow orchestration)
    role.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          'states:StartExecution',
          'states:DescribeExecution',
          'states:StopExecution',
          'states:GetExecutionHistory'
        ],
        resources: ['*'],
      })
    );

    // Add permissions for EventBridge (for scheduling and event-driven workflows)
    role.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          'events:PutEvents',
          'events:PutRule',
          'events:PutTargets',
          'events:DeleteRule',
          'events:RemoveTargets'
        ],
        resources: ['*'],
      })
    );

    return role;
  }
}
