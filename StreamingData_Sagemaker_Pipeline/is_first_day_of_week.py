import logging
import os
import pathlib
import subprocess
import sys
print(sys.version)
import json
subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn==1.2.2", "hopsworks==3.4.3", "pandas==1.5.3", "joblib==1.3.2", "wandb==0.16.1", "evidently==0.4.14", "pyyaml", "boto3","pymysql==1.1.0","psycopg2-binary==2.9.9","xgboost==2.0.2"])
import numpy as np
import boto3
import yaml
import psycopg2
import pandas as pd




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

def create_postgres_connection(config):
    """
    Establish a connection to the PostgreSQL database.
    
    Parameters:
    config: YAML file with Postgres Credentials

    Returns: 
    A connection object to the PostgreSQL database
    """
    user = config['postgres']['user']
    password = config['postgres']['password']
    host = config['postgres']['host']
    database = config['postgres']['database']
    port = int(config['postgres']['port'])

    try:
        connection = psycopg2.connect(
            user=user,
            password=password,
            host=host,
            database=database,
            port=port
        )
        
    except Exception as e:
        print(f"An error occurred while connecting to Postgres: {str(e)}")

    else:
        return connection



def read_data(connection, table_name):
    '''
    Read data from a MySQL/Postgres table.

    Parameters:
    - connection: MySQL/Postgres connection object.
    - table_name (str): Name of the table to fetch data from.

    Returns:
    - df (pd.DataFrame): DataFrame containing the fetched data.
    '''
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, connection)
    return df






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

    # Creating connection to Postgres and MySQL Databases
    postgres_connection = create_postgres_connection(config)

    logger.info("Created connection to Postgres Databases")

    # Fetch data from PostgreSQL tables
    constants = read_data(postgres_connection, "constants")

    # Initialize an empty dictionary to store the evaluation results
    evaluation_results = {}

    # Check if it's the first day of a new week
    if constants.at[0,'is_first_day_of_new_week'] == True:
        evaluation_results["is_first_day_of_new_week"] = 1
    else:
        evaluation_results["is_first_day_of_new_week"] = 0

    # Set the output directory for the evaluation results
    output_dir = "/opt/ml/processing/evaluation1"

    # Create the output directory if it doesn't exist or if it's empty
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Set the path for the evaluation results file
    evaluation_path = f"{output_dir}/evaluation1.json"

    # Write the evaluation results to a JSON file
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(evaluation_results))
