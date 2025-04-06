import logging
import os
import pathlib
import subprocess
import sys
import json
subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn==1.2.2", "hopsworks==3.4.3", "pandas==1.5.3", "joblib==1.3.2", "wandb==0.16.1", "evidently==0.4.14", "pyyaml", "boto3","pymysql==1.1.0","psycopg2-binary==2.9.9","xgboost==2.0.2"])

import numpy as np
import boto3
import yaml
import hopsworks
import psycopg2
import pymysql
import pandas as pd
from evidently.test_suite import TestSuite
from evidently.tests.base_test import generate_column_tests
from evidently.tests import *
from datetime import datetime, timedelta

from evidently.metrics.base_metric import generate_column_metrics
from evidently.metrics import *
from sklearn.model_selection import train_test_split

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



def create_mysql_connection(config):
    '''
    Create a MySQL connection.

    Parameters:
    config: YAML file with Postgres Credentials

    Returns:
    - connection (pymysql.connections.Connection): MySQL connection object.
    '''
    try:
        host = config['mysql']['host']
        user = config['mysql']['user']
        password = config['mysql']['password']
        database = config['mysql']['database']

        # Establish a connection to the MySQL database
        connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        print("Connected to MySQL database successfully!")
        
    except Exception as e:
        print(f"An error occurred while connecting to Mysql: {str(e)}")
        return None
    
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

def run_data_quality_check(data_name, reference_data, current_data, timestamp):
    """
    Run data quality checks on the given reference and current data.

    This function performs data quality checks using a TestSuite object,
    which includes the following tests:
    - TestNumberOfRowsWithMissingValues: Checks for the number of rows with missing values.
    - TestNumberOfDuplicatedRows: Checks for the number of duplicated rows.
    - TestNumberOfDuplicatedColumns: Checks for the number of duplicated columns.
    - TestColumnsType: Checks if the columns' types match between reference and current data.

    The function then saves the test results as an HTML file and uploads it to an S3 bucket.

    Parameters:
    data_name (str): The name of the data to be checked.
    reference_data (pandas.DataFrame): The reference data for comparison.
    current_data (pandas.DataFrame): The current data to be checked.
    timestamp (str): The timestamp for the report filename.

    Returns:
    tuple: A tuple containing the TestSuite object and its JSON representation.
    """
    tests = TestSuite(tests=[
        TestNumberOfRowsWithMissingValues(),
        TestNumberOfDuplicatedRows(),
        TestNumberOfDuplicatedColumns(),
        TestColumnsType()
    ])

    tests.run(reference_data=reference_data, current_data=current_data)
    
    s3 = boto3.client('s3')
    tests.save_html(base_dir+data_name+"_results.html")

    s3.upload_file(base_dir+data_name+"_results.html", "truck-eta-classification-logs", "reports/"+timestamp+"/"+data_name+"_results.html")

    return tests, tests.json()


def fetch_data(config, feature_group_name):
    """
    Fetch data from a feature group in Hopsworks.
    
    Parameters:
    config (dict): The configuration as a dictionary.
    
    Returns:
    pandas.DataFrame: The data as a pandas DataFrame.
    """

    # Login to hopsworks by entering API key value
    project = hopsworks.login(api_key_value=config['hopsworks']['api_key'])

    # Get the feature store
    fs = project.get_feature_store()

    try:
        # Retrieve feature group
        feature_group = fs.get_feature_group(
            feature_group_name,
            version=config['hopsworks']['feature_group_version']
        )

        # Select the query
        query = feature_group.select_all()

        # Read all data
        data = query.read(read_options={"use_hive": config['hopsworks']['use_hive_option']})

        return data

    except Exception as e:
        # Log the error
        logger.error(f"An error occurred during data fetching: {str(e)}")

def fix_datasets(dataframe, result_json):
    """
    Fix a dataset based on the results of a data quality check.

    This function modifies the input dataframe by removing rows or columns with issues,
    and changing column types as necessary. It takes a JSON string that contains the
    results of a data quality check and applies the necessary changes to the dataframe.

    Args:
        dataframe (pandas.DataFrame): The input dataframe to be fixed.
        result_json (str): The JSON string containing the results of the data quality check.

    Returns:
        pandas.DataFrame: The fixed dataframe with all identified issues resolved.
    """
    # Copy the dataframe to avoid modifying the original
    fixed_df = dataframe.copy()
    # Convert the JSON string to a dictionary
    result_json = json.loads(result_json)


    # Loop through the tests in the result_json
    for test in result_json['tests']:
        if test['status'] == 'FAIL':
            # Identify the failed test and take corrective actions
            test_name = test['name']

            if test_name == 'The Number Of Rows With Missing Values':
                # Remove rows with missing values
                fixed_df = fixed_df.dropna()

            elif test_name == 'Number of Duplicate Rows':
                # Remove duplicate rows
                fixed_df = fixed_df.drop_duplicates()

            elif test_name == 'Number of Duplicate Columns':
                # Remove duplicate columns
                duplicate_columns = test['parameters']['value']
                fixed_df = fixed_df.loc[:, ~fixed_df.columns.duplicated()]

            elif test_name == 'Column Types':
                # Change column types as per the expected types in the test
                for column_info in test['parameters']['columns']:
                    column_name = column_info['column_name']
                    expected_type = column_info['expected_type']
                    if expected_type == 'object_':
                        fixed_df[column_name] = fixed_df[column_name].astype('object')
                    else:
                      # Convert the column to the expected type
                        fixed_df[column_name] = fixed_df[column_name].astype(expected_type)

    return fixed_df

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
    
    # Set the wandb API key from the config
    os.environ['WANDB_API_KEY'] = config['wandb']['wandb_api_key']

    # Connect to Postgres and MySQL databases
    postgres_connection = create_postgres_connection(config)
    mysql_connection = create_mysql_connection(config)
    logger.info("Created connection to Postgres and MySQL Databases")

    # Fetch data from PostgreSQL and MySQL tables
    row_counts_df = read_data(postgres_connection, "constants")
    truck_schedule_data = read_data(mysql_connection, config['table_names']['schedule_data'])
    logger.info("Read data from Postgres and MySQL databases.")

    # Set the timestamp for the current date
    timestamp = datetime.now().strftime("%m-%d-%Y")

    # Check if the delay column of the truck schedule table is updated in the system
    max_date = row_counts_df['day']
    if str(max_date[0]) >= "2019-02-17":
        previous = max_date[0] - timedelta(days=2)
        logger.info("Delay column of truck schedule table for the date %s is updated in system!", str(previous))

        # Set the lower and upper dates for filtering data
        lower_date = str(previous) + " 00:00:00"
        upper_date = str(previous) + " 23:59:59"
        logger.info("Lower date for updation: %s", lower_date)
        logger.info("Upper date for updation: %s", upper_date)

        # Login to Hopsworks using the API key value
        project = hopsworks.login(api_key_value="YxM3sRpZPNNNeEdJ.TJnkihXVHEwK4XiX2WLYnFIe04D5nHCWj6rGdsBCJp5ecP7ocPvA8gdX2PfUh0vR")

        # Get the feature store
        fs = project.get_feature_store()

        # Fetch the truck schedule details feature group
        truck_schedule_df = fetch_data(config, 'truck_schedule_details_fg')

        # Convert the estimated_arrival column to datetime
        truck_schedule_data['estimated_arrival'] = pd.to_datetime(truck_schedule_data['estimated_arrival'])

        # Filter truck_schedule_data based on the estimated_arrival within the lower and upper dates
        truck_schedule_data = truck_schedule_data[(truck_schedule_data['estimated_arrival'] >= lower_date) & (truck_schedule_data['estimated_arrival'] <= upper_date)]
        truck_schedule_data.reset_index(drop=True, inplace=True)

        # Run data quality checks on the truck schedule data
        truck_schedule_delay_quality_dashboard, truck_schedule_delay_quality_results = run_data_quality_check('Truck_Schedule_Streaming_Delay_Updation', truck_schedule_df.dropna(subset=['delay']), truck_schedule_data, timestamp)

        # Fix any issues in the datasets
        truck_schedule_data = fix_datasets(truck_schedule_data, truck_schedule_delay_quality_results)

        # Convert the delay column to Int64
        truck_schedule_data['delay'] = truck_schedule_data['delay'].astype('Int64')


        # Check if there are any rows in the truck_schedule_data DataFrame
        if truck_schedule_data.shape[0] > 0:
            # Get the truck_schedule_details_fg feature group from Hopsworks
            schedule_df_fg = fs.get_feature_group('truck_schedule_details_fg', version=1) 

            # Convert the estimated_arrival column to datetime format
            truck_schedule_data['estimated_arrival'] = pd.to_datetime(truck_schedule_data['estimated_arrival'])  

            # Insert the updated truck_schedule_data into the truck_schedule_details_fg feature group
            schedule_df_fg.insert(truck_schedule_data)  

            # Log that the insert operation is completed
            logger.info("truck_schedule_details_fg done")  

            # Log the number of rows updated in the feature group
            logger.info("Total number of %s rows updated in truck_schedule_details_fg feature group in hopsworks.", str(truck_schedule_data.shape[0]))  

            # Set the email subject and message
            subject = 'Notification from SageMaker Streaming Pipeline'  
            message = 'Total number of ' + str(truck_schedule_data.shape[0]) + ' rows updated in truck_schedule_details_fg feature group in hopsworks.'  
            
            # Send the email notification
            send_email(subject, message)  
                    
            
        # Fetch the final_data feature group from Hopsworks
        final_merge = fetch_data(config,'final_data')

        # Filter final_merge based on the estimated_arrival column within the lower and upper dates and drop the delay column
        final_merge_filtered = final_merge[(final_merge['estimated_arrival']>=lower_date) & (final_merge['estimated_arrival']<=upper_date)].drop("delay", axis=1)

        # Log the number of rows in final_merge_filtered
        logger.info(final_merge_filtered.shape[0])

        # Log the number of rows in truck_schedule_data
        logger.info(truck_schedule_data.shape[0])



        if final_merge_filtered.shape[0] == truck_schedule_data.shape[0]:
            # Merge final_merge_filtered and truck_schedule_data on truck_id, route_id, departure_date, and estimated_arrival
            final_merge_filtered = final_merge_filtered.merge(truck_schedule_data, on=['truck_id', 'route_id', 'departure_date', 'estimated_arrival'], how='left')
            
            # Print information about the final_merge DataFrame
            print(final_merge.info())

            # Change the delay column type to np.float64 in final_merge
            final_merge['delay'] = final_merge['delay'].astype(np.float64)
            print(final_merge.info())
            
            # Change the delay column type to np.float64 in final_merge_filtered
            final_merge_filtered['delay'] = final_merge_filtered['delay'].astype(np.float64)

            if final_merge_filtered.shape[0] > 0:
                # Print information about the final_merge and final_merge_filtered DataFrames
                print("Final_merge columns - ", final_merge.columns)
                print("final_merge_filtered columns - ", final_merge_filtered.columns)
                print("Final_merge shape - ", final_merge.shape)
                print("final_merge_filtered shape - ", final_merge_filtered.shape)

                # Run data quality checks and get the results
                final_merge_updation_quality_dashboard, final_merge_updation_quality_results = run_data_quality_check('Final_Merge_Delay_Updation', final_merge.dropna(subset=['delay']), final_merge_filtered, timestamp)

                # Parse the JSON-formatted final_merge_updation_quality_results
                final_merge_updation_quality_results = json.loads(final_merge_updation_quality_results)

                # If the test for column types fails
                if final_merge_updation_quality_results['tests'][3]['status'] == 'FAIL':
                    # Loop through the columns with failed tests
                    for column_info in final_merge_updation_quality_results['tests'][3]['parameters']['columns']:
                        # Get the column name and expected type
                        column_name = column_info['column_name']
                        expected_type = column_info['expected_type']
                        # If the expected type is 'object\_'
                        if expected_type == 'object_':
                            # Change the column type to 'object' in final_merge_filtered
                            final_merge_filtered[column_name] = final_merge_filtered[column_name].astype('object')
                        else:
                            # Convert the column to the expected type in final_merge_filtered
                            final_merge_filtered[column_name] = final_merge_filtered[column_name].astype(expected_type)

                # Change the delay column type to 'Int64' in final_merge_filtered
                final_merge_filtered['delay'] = final_merge_filtered['delay'].astype('Int64')

                # Insert the updated final_merge_filtered into the final_data feature group in Hopsworks
                final_merge_fg = fs.get_feature_group('final_data', version=1)
                final_merge_fg.insert(final_merge_filtered)

                logger.info("Total number of %s rows updated in final_merge feature group in hopsworks.", str(final_merge_filtered.shape[0]))

                # Send email notification with the number of rows updated in the final_merge feature group
                subject = 'Notification from SageMaker Streaming Pipeline'
                message = 'Total number of ' + str(final_merge_filtered.shape[0]) + ' rows updated in final_merge feature group in hopsworks.'
                send_email(subject, message)

        else:
            # Send email notification with the number of rows not matched in the final merge
            subject = 'Notification from SageMaker Streaming Pipeline'
            message = 'Number of rows were not matched in final merge. ' + str(final_merge_filtered.shape[0]) +" "+str(truck_schedule_data.shape[0])
            send_email(subject, message)
    
    # Set the base directory for saving the preprocessed data
    base_dir = "/opt/ml/processing"

    # Create a dictionary with the result of the preprocessing
    dict_ = {'Result': 'Done'}

    # Save the dictionary as a DataFrame to a CSV file
    pd.DataFrame([dict_]).to_csv(f"{base_dir}/result1/result1.csv", header=True, index=False)