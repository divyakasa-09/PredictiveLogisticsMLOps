import boto3
import os
import tempfile
from botocore.exceptions import ClientError

def lambda_handler(event, context):
    # Set your CloudWatch Log Group and S3 Bucket details
    log_group_name = '/aws/sagemaker/ProcessingJobs'
    s3_bucket_name = ''
    s3_object_key = ''

    # Create a CloudWatch Logs client
    cloudwatch_logs = boto3.client('logs')

    try:
        # Get the latest log stream name
        response = cloudwatch_logs.describe_log_streams(
            logGroupName=log_group_name,
            orderBy='LastEventTime',
            descending=True,
            limit=1
        )

        latest_log_stream = response['logStreams'][0]['logStreamName']

        # Get the logs from the latest log stream
        log_events = cloudwatch_logs.get_log_events(
            logGroupName=log_group_name,
            logStreamName=latest_log_stream,
            limit=10000  # Adjust based on your requirements
        )['events']

        # Create a temporary file to store logs
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            for log_event in log_events:
                temp_file.write(log_event['message'] + '\n')

        # Upload the log file to S3
        s3 = boto3.client('s3')
        s3.upload_file(temp_file.name, s3_bucket_name, s3_object_key)

        # Get a pre-signed URL for the uploaded file
        url = s3.generate_presigned_url(
            ClientMethod='get_object',
            Params={'Bucket': s3_bucket_name, 'Key': s3_object_key},
            ExpiresIn=3600  # Adjust the expiration time as needed
        )

        return {
            'statusCode': 200,
            'body': {'logData':str(url),
                    'pipelineArn': event['detail']['pipelineArn'],
                    'executionStartTime': event['detail']['executionStartTime'],
                    'executionEndTime': event['detail']['executionEndTime']
            }
        }

    except ClientError as e:
        return {
            'statusCode': 500,
            'body': f'Error: {str(e)}'
        }
