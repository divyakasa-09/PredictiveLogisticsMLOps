import logging
import os
import pathlib
import subprocess
import sys
print(sys.version)
subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn==1.2.2", "pandas==1.5.3", "joblib==1.3.2","pyyaml", "boto3"])
import boto3
import yaml



logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def load_config(file_path):
    """
    Load configuration from a YAML file.
    
    Parameters:
    file_path (str): The path to the YAML file. Default is 'config.yaml'.
    
    Returns:
    dict: The configuration as a dictionary.
    """
    with open(file_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config



def send_email(subject, message):
    """
    Send an email using Amazon SNS (Simple Notification Service).

    This function sends an email using the Amazon SNS service, which requires
    AWS credentials and permissions to publish to a specified topic.

    Args:
        subject (str): The email subject.
        message (str): The email message body.

    Returns:
        dict: A dictionary containing the response from the SNS service,
            including the MessageId of the published message.
    """
    sns_client = boto3.client('sns',region_name='us-east-2')
    
    topic_arn = 'arn:aws:sns:us-east-2:143176219551:StreamingNotification'

    response = sns_client.publish(
        TopicArn=topic_arn,
        Subject=subject,
        Message=message
    )
    print(f"Email sent! MessageId: {response['MessageId']}")
    
    


if __name__ == "__main__":
    
    import argparse

    # Create an ArgumentParser object to parse command line arguments
    parser = argparse.ArgumentParser()

    # Define the required --config-file-data argument
    parser.add_argument("--config-file-data", type=str, required=True, help="Path to the config data")

    # Parse the arguments
    args = parser.parse_args()

    # Initialize the logger
    logger.debug("Starting fetching data")

    # Set the base directory for the processing pipeline
    base_dir = "/opt/ml/processing"

    # Create the data folder if it doesn't exist
    data_folder = pathlib.Path(f"{base_dir}/data")
    data_folder.mkdir(parents=True, exist_ok=True)

    # Get the config file path from the parsed arguments
    config_file = args.config_file_data

    # Extract the bucket and key from the config file path
    bucket = config_file.split("/")[2]
    key = "/".join(config_file.split("/")[3:])

    # Print the bucket and key
    print(bucket, key)

    # Log that the data is being downloaded
    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)

    # Define the file name for the downloaded config file
    fn = f"{base_dir}/data/config.yaml"

    # Print the file name
    print(fn)

    # Download the config file from the S3 bucket
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)

    # Print the file name again
    print(fn)

    # Load the config file into memory
    config = load_config(fn)

    # Remove the downloaded config file
    os.unlink(fn)
    
    subject = 'Notification from SageMaker Streaming Pipeline'
    message = 'Today is not the first day of the week. Pipeline ended.'
    send_email(subject, message)