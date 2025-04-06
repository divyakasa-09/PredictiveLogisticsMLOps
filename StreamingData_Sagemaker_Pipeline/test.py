import boto3

def send_email(subject, message):
    sns_client = boto3.client('sns',region_name='us-east-2')
    
    topic_arn = 'arn:aws:sns:us-east-2:143176219551:new-Streaming-pipeline-notification'

    response = sns_client.publish(
        TopicArn=topic_arn,
        Subject=subject,
        Message=message
    )
    print(f"Email sent! MessageId: {response['MessageId']}")
    

    
subject = "Notification from SageMaker Streaming Pipeline"
message = "Testing successful"
send_email(subject, message)