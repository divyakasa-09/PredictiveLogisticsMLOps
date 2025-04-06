#!/usr/bin/env python

from aws_cdk import (
    App,
    Stack,
    aws_events as events,
    aws_events_targets as targets,
    aws_kms as kms,
    aws_sns as sns,
    aws_sns_subscriptions as subscriptions,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as tasks,
    aws_iam as iam,
)
from aws_cdk import aws_lambda as _lambda

from constructs import Construct


class sagemaker_pipeline_notifications(Stack):
    def __init__(
        self, scope: Construct, id: str, notification_email: str, **kwargs
    ) -> None:
        super().__init__(scope, id, **kwargs)

        # Create SNS Topic for notifications of failed SageMaker Pipeline executions
        sns_key = kms.Alias.from_alias_name(self, "SnsKey", "alias/aws/sns")
        sagemaker_pipeline_notifications_topic = sns.Topic(
            self,
            "SageMakerPipelineFailedNotification",
            master_key=sns_key,
        )
        # Subscribe to notifications
        sagemaker_pipeline_notifications_topic.add_subscription(
            subscriptions.EmailSubscription(notification_email)
        )

        # AWS Step Function Definition
        
        # Create IAM Role for Lambda Function
        lambda_role = iam.Role(
            self,
            "LambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description="Role for Lambda to access CloudWatch Logs"
        )
        
        # Attach CloudWatch Logs Access Policy
        lambda_role.add_to_policy(iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
                "logs:DescribeLogStreams",
                "logs:GetLogEvents",
                "s3:GetObject",
                "s3:PutObject",
                "s3:ListBucket"
            ],
            resources=[
                "arn:aws:logs:*:*:*",
                "arn:aws:s3:::*"
            ]
        ))
        
        # Define Lambda function
        fetch_logs_lambda = _lambda.Function(
            self,
            "FetchLogsFunction",
            runtime=_lambda.Runtime.PYTHON_3_8,
            handler="lambda_function.lambda_handler",
            code=_lambda.Code.from_asset("lambda_function.zip"),
            role=lambda_role
        )
        
        # Add Lambda step in Step Functions for failure
        fetch_logs_step_failure = tasks.LambdaInvoke(
            self,
            "Fetch CloudWatch Logs For Failure",
            lambda_function=fetch_logs_lambda,
            output_path="$.Payload",
        )
        
        # Add Lambda step in Step Functions for success
        fetch_logs_step_success = tasks.LambdaInvoke(
            self,
            "Fetch CloudWatch Logs For Success",
            lambda_function=fetch_logs_lambda,
            output_path="$.Payload",
        )
        
        # Step to notify of a failed SageMaker Pipeline execution via SNS Topic.
        notify_failed_sagemaker_pipeline_execution = tasks.SnsPublish(
            self,
            "Notify that a SageMaker Pipeline execution has failed",
            topic=sagemaker_pipeline_notifications_topic,
            integration_pattern=sfn.IntegrationPattern.REQUEST_RESPONSE,
            subject=sfn.JsonPath.format(
                "Amazon SageMaker Pipeline Failed - Pipeline Name: {}",
                sfn.JsonPath.array_get_item(
                    sfn.JsonPath.string_split(
                        sfn.JsonPath.string_at("$.body.pipelineArn"), "/"
                    ),
                    1,
                ),
            ),
            message=sfn.TaskInput.from_text(
                sfn.JsonPath.format(
                    "The SageMaker Pipeline, {}, started at {} and failed at {}. Click on this link to Download the log file: {}",
                    sfn.JsonPath.array_get_item(
                        sfn.JsonPath.string_split(
                            sfn.JsonPath.string_at("$.body.pipelineArn"), "/"
                        ),
                        1,
                    ),
                    sfn.JsonPath.string_at("$.body.executionStartTime"),
                    sfn.JsonPath.string_at("$.body.executionEndTime"),
                    sfn.JsonPath.string_at("$.body.logData"),
                )
            ),
        )

        sagemaker_pipeline_notifications_state_machine = sfn.StateMachine(
            self,
            "SageMakerPipelineNotificationsStateMachine",
            state_machine_name="sagemaker-pipeline-notifications-python-failed",
            definition = sfn.Chain.start(fetch_logs_step_failure).next(notify_failed_sagemaker_pipeline_execution),
        )

        # EventBridge rule that triggers the state machine whenever a SageMaker Pipeline execution fails.
        events.Rule(
            self,
            "SageMakerPipelineExecutionFailedTrigger",
            event_pattern=events.EventPattern(
                source=["aws.sagemaker"],
                detail={"currentPipelineExecutionStatus": ["Failed"]},
                detail_type=[
                    "SageMaker Model Building Pipeline Execution Status Change"
                ],
            ),
            targets=[
                targets.SfnStateMachine(sagemaker_pipeline_notifications_state_machine)
            ],
        )
        
        
        # Step to notify of a Succeeded SageMaker Pipeline execution via SNS Topic.
        notify_successful_sagemaker_pipeline_execution = tasks.SnsPublish(
            self,
            "Notify that a SageMaker Pipeline execution has succeeded",
            topic=sagemaker_pipeline_notifications_topic,
            integration_pattern=sfn.IntegrationPattern.REQUEST_RESPONSE,
            subject=sfn.JsonPath.format(
                "Amazon SageMaker Pipeline Succeeded - Pipeline Name: {}",
                sfn.JsonPath.array_get_item(
                    sfn.JsonPath.string_split(
                        sfn.JsonPath.string_at("$.body.pipelineArn"), "/"
                    ),
                    1,
                ),
            ),
            message=sfn.TaskInput.from_text(
                sfn.JsonPath.format(
                    "The SageMaker Pipeline, {}, started at {} and succeeded at {}. Click on this link to Download the log file: {}",
                    sfn.JsonPath.array_get_item(
                        sfn.JsonPath.string_split(
                            sfn.JsonPath.string_at("$.body.pipelineArn"), "/"
                        ),
                        1,
                    ),
                    sfn.JsonPath.string_at("$.body.executionStartTime"),
                    sfn.JsonPath.string_at("$.body.executionEndTime"),
                    sfn.JsonPath.string_at("$.body.logData"),
                )
            ),
        )
        
        sagemaker_pipeline_notifications_state_machine_success = sfn.StateMachine(
            self,
            "SageMakerPipelineNotificationsStateMachineSuccess",
            state_machine_name="sagemaker-pipeline-notifications-python-succeed",
            definition=sfn.Chain.start(fetch_logs_step_success).next(notify_successful_sagemaker_pipeline_execution),
        )
        
        # EventBridge rule that triggers the state machine whenever a SageMaker Pipeline execution succeeded.
        events.Rule(
            self,
            "SageMakerPipelineExecutionSuccessTrigger",
            event_pattern=events.EventPattern(
                source=["aws.sagemaker"],
                detail_type=[
                    "SageMaker Model Building Pipeline Execution Status Change"
                ],
                detail={"currentPipelineExecutionStatus": ["Succeeded"]},
            ),
            targets=[
                targets.SfnStateMachine(sagemaker_pipeline_notifications_state_machine_success)
            ],
        )


app = App()

sagemaker_pipeline_notifications(
    app,
    "SageMakerPipelineNotificationsPython",
    notification_email="",
)
app.synth()