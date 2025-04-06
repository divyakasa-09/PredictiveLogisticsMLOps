import logging
import os
import pathlib
import subprocess
import sys
print(sys.version)
import json
#subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3","hopsworks==3.4.3", "pyyaml","typing-extensions==4.3.0","sqlalchemy==1.4.39"])

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
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset
from evidently.tests import *
from datetime import datetime, timedelta
import time
from evidently import ColumnMapping

from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import *
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import wandb
import xgboost as xgb
import joblib


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
    tests = TestSuite(tests=[
   TestNumberOfRowsWithMissingValues(),
    TestNumberOfDuplicatedRows(),
    TestNumberOfDuplicatedColumns(),
    TestColumnsType()
    ])
    tests.run(reference_data=reference_data, current_data=current_data)
    #tests.save_html(data_name+"_results.html")
    subject = 'Notification from SageMaker Streaming Pipeline'
    message = tests.json()
    #send_email(subject, message)
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
    #project = hopsworks.login(api_key_value=config['hopsworks']['api_key'])
    project = hopsworks.login(api_key_value="YxM3sRpZPNNNeEdJ.TJnkihXVHEwK4XiX2WLYnFIe04D5nHCWj6rGdsBCJp5ecP7ocPvA8gdX2PfUh0vR")

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

def return_passed_rows(data_result_list):
        '''
        Return the rows which passed
        '''

        fixed_dataframes = []
        for data_tuple in data_result_list:
            fixed_dataframes.append(fix_datasets(data_tuple[0], data_tuple[1]))

        new_routes_weather_df = fixed_dataframes[0]
        new_truck_schedule_df= fixed_dataframes[1]
        new_traffic_details_df= fixed_dataframes[2]
        new_city_weather_df= fixed_dataframes[3]
        
        return new_routes_weather_df, new_truck_schedule_df, new_traffic_details_df, new_city_weather_df

        
def send_email(subject, message):
    sns_client = boto3.client('sns',region_name='us-east-2')
    
    topic_arn = 'arn:aws:sns:us-east-2:143176219551:StreamingNotification'

    response = sns_client.publish(
        TopicArn=topic_arn,
        Subject=subject,
        Message=message
    )
    print(f"Email sent! MessageId: {response['MessageId']}")
    

def send_new_data_to_hopsworks(new_routes_weather_df, new_truck_schedule_df, new_traffic_details_df, new_city_weather_df, config):
    total_new_rows = new_routes_weather_df.shape[0]+new_truck_schedule_df.shape[0]+new_traffic_details_df.shape[0]+new_city_weather_df.shape[0]
        
    if total_new_rows==0:
        subject = 'Notification from SageMaker Streaming Pipeline'
        message = 'There was no data passed from data checks.'
        send_email(subject, message)
    else:
        logger.info("New rows:")
        logger.info(total_new_rows)

        # Login to hopsworks by entering API key value
        #project = hopsworks.login(api_key_value=config['hopsworks']['api_key'])
        project = hopsworks.login(api_key_value="YxM3sRpZPNNNeEdJ.TJnkihXVHEwK4XiX2WLYnFIe04D5nHCWj6rGdsBCJp5ecP7ocPvA8gdX2PfUh0vR")

        # Get the feature store
        fs = project.get_feature_store()

        logger.info("Successfully logged into hopsworks")

        if new_routes_weather_df.shape[0]>0:
            route_weather_fg = fs.get_feature_group('route_weather_details_fg', version=1)
            route_weather_fg.insert(new_routes_weather_df)
            logger.info("route_weather_details_fg done")

        if new_truck_schedule_df.shape[0]>0:
            schedule_df_fg = fs.get_feature_group('truck_schedule_details_fg', version=1)
            new_truck_schedule_df['estimated_arrival']=pd.to_datetime(new_truck_schedule_df['estimated_arrival'])
            schedule_df_fg.insert(new_truck_schedule_df)
            logger.info("truck_schedule_details_fg done")

        if new_traffic_details_df.shape[0]>0:
            traffic_df_fg = fs.get_feature_group('traffic_details_fg', version=1)
            new_traffic_details_df['date']=pd.to_datetime(new_traffic_details_df['date'])
            traffic_df_fg.insert(new_traffic_details_df)
            logger.info("traffic_details_fg done")

        if new_city_weather_df.shape[0]>0:
            city_weather_df_fg = fs.get_feature_group('city_weather_details_fg', version=1)
            new_city_weather_df['date']=pd.to_datetime(new_city_weather_df['date'])
            city_weather_df_fg.insert(new_city_weather_df)
            logger.info("city_weather_details_fg done")
            
        subject = 'Notification from SageMaker Streaming Pipeline'
        message = 'New data detected! \n'+str(new_routes_weather_df.shape[0])+' rows added in route_weather_details_fg feature group.\n'+str(new_truck_schedule_df.shape[0])+' rows added in truck_schedule_details_fg feature group. \n'+str(new_traffic_details_df.shape[0])+' rows added in traffic_details_fg feature group. \n'+str(new_city_weather_df.shape[0])+' rows added in city_weather_details_fg feature group.'
        
        send_email(subject, message)

        logger.info("All done")


if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file-data", type=str, required=True, help="Path to the config data")
    args = parser.parse_args()
    
    logger.debug("Starting fetching data")
    
    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    
    config_file = args.config_file_data
    
    bucket = config_file.split("/")[2]
    key = "/".join(config_file.split("/")[3:])
    
    print(bucket, key)
    
    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/config.yaml"
    
    print(fn)
    
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)
    
    print(fn)
    config = load_config(fn)
    
    os.unlink(fn)
    
    # WANDB
    os.environ['WANDB_API_KEY'] = config['wandb']['wandb_api_key']

    timestamp = datetime.now().strftime("%m-%d-%Y")

    # Creating connection to Postgres and MySQL Databases
    postgres_connection = create_postgres_connection(config)
    mysql_connection = create_mysql_connection(config)
    
    logger.info("Created connection to Postgres and MySQL Databases")
    
    # Fetch data from PostgreSQL tables
    row_counts_df = read_data(postgres_connection, "constants")

    routes_weather_df = read_data(postgres_connection, config['table_names']['route_weather'])

    # Fetch data from MySQL tables
    traffic_df = read_data(mysql_connection, config['table_names']['traffic_data'])
    truck_schedule_df = read_data(mysql_connection, config['table_names']['schedule_data'])
    truck_schedule_data = read_data(mysql_connection, config['table_names']['schedule_data'])

    city_weather_df = read_data(mysql_connection, config['table_names']['weather_data'])
    
    routes_weather_df_row = routes_weather_df.shape[0]
    traffic_df_row = traffic_df.shape[0]
    truck_schedule_df_row = truck_schedule_df.shape[0]
    city_weather_df_row = city_weather_df.shape[0]
    
    print(routes_weather_df.shape[0]-int(row_counts_df['routes_weather'][0]))
    print(traffic_df.shape[0]-int(row_counts_df['traffic_details'][0]))
    print(city_weather_df.shape[0]-int(row_counts_df['city_weather'][0]))
    
    new_routes_weather_df = routes_weather_df.tail(routes_weather_df.shape[0]-int(row_counts_df['routes_weather'][0]))
    new_truck_schedule_df = truck_schedule_df.tail(truck_schedule_df.shape[0]-int(row_counts_df['truck_schedule_data'][0]))
    new_traffic_details_df = traffic_df.tail(traffic_df.shape[0]-int(row_counts_df['traffic_details'][0]))
    new_city_weather_df = city_weather_df.tail(city_weather_df.shape[0]-int(row_counts_df['city_weather'][0]))
    
    print(new_routes_weather_df.shape[0],new_truck_schedule_df.shape[0],new_traffic_details_df.shape[0],new_city_weather_df.shape[0])
    
    total_new_rows = new_routes_weather_df.shape[0]+new_truck_schedule_df.shape[0]+new_traffic_details_df.shape[0]+new_city_weather_df.shape[0]
    
    if_new_data_available = False

    if total_new_rows==0:
        subject = 'Notification from SageMaker Streaming Pipeline'
        message = 'There was no incoming streaming data.'
        send_email(subject, message)
    else:
        logger.info("Performing data checks...")
        
        # Fetching old data from hopsworks
        routes_weather_df = fetch_data(config, 'route_weather_details_fg')
        truck_schedule_df = fetch_data(config, 'truck_schedule_details_fg')
        traffic_df = fetch_data(config, 'traffic_details_fg')
        city_weather_df = fetch_data(config, 'city_weather_details_fg')
        
        # Data Checks and Report
        
        logger.info("Data Checks and Reports...")
        
        new_routes_weather_df.rename(columns = {'Date':'date'}, inplace = True) 

        routes_weather_quality_dashboard, routes_weather_quality_results = run_data_quality_check('Route_Weather_Streaming',routes_weather_df, new_routes_weather_df, timestamp)
        truck_schedule_quality_dashboard, truck_schedule_quality_results = run_data_quality_check('Truck_Schedule_Streaming',truck_schedule_df.drop('delay',axis=1), new_truck_schedule_df.drop('delay',axis=1), timestamp)
        traffic_quality_dashboard, traffic_quality_results = run_data_quality_check('Traffic_Details_Streaming',traffic_df, new_traffic_details_df, timestamp)
        city_weather_quality_dashboard, city_weather_quality_results = run_data_quality_check('City_Weather_Streaming',city_weather_df, new_city_weather_df, timestamp)
        
        logger.info("Checking if the data passed the tests...")

        # 4. Check if the data passed the tests
        all_results = [routes_weather_quality_results, truck_schedule_quality_results, traffic_quality_results, city_weather_quality_results]

        data_result_list = [
            (new_routes_weather_df, routes_weather_quality_results),
            (new_truck_schedule_df, truck_schedule_quality_results),
            (new_traffic_details_df, traffic_quality_results),
            (new_city_weather_df, city_weather_quality_results)
        ]

        # Check if all checks passed
        # if all results check is pass then go ahead otherwise run return_passed_rows function and fix datasets 
        # Using map and json.loads
        json_result_list = list(map(json.loads, all_results))
        if all(result['tests'][i]['status'] == 'SUCCESS' for result in json_result_list for i in range(len(result['tests']))):
            logger.info("All data quality checks passed.")
            # Send new data to hopsworks
            send_new_data_to_hopsworks(new_routes_weather_df, new_truck_schedule_df, new_traffic_details_df, new_city_weather_df, config)
            if_new_data_available = True
        elif all(result['tests'][i]['status'] == 'FAIL' for result in json_result_list for i in range(len(result['tests']))):
            logger.info("All data quality checks failed.")
            subject = 'Notification from SageMaker Streaming Pipeline'
            message = 'All data quality checks failed!'
            send_email(subject, message)
        else:
            logger.info("Some data quality checks failed. Running Returning Partial Passed Rows Function!")
            subject = 'Notification from SageMaker Streaming Pipeline'
            message = 'Some data quality checks failed! Returning partially passed rows.'
            send_email(subject, message)
            
            print(new_routes_weather_df.shape[0],new_truck_schedule_df.shape[0],new_traffic_details_df.shape[0],new_city_weather_df.shape[0])

            new_routes_weather_df, new_truck_schedule_df, new_traffic_details_df, new_city_weather_df = return_passed_rows(data_result_list)
            
            new_truck_schedule_df['delay'] = new_truck_schedule_df['delay'].astype('Int64')
            
            print(new_routes_weather_df.shape[0],new_truck_schedule_df.shape[0],new_traffic_details_df.shape[0],new_city_weather_df.shape[0])

            # Send new data to hopsworks
            send_new_data_to_hopsworks(new_routes_weather_df, new_truck_schedule_df, new_traffic_details_df, new_city_weather_df, config)
            
            logger.info(new_truck_schedule_df.shape)
            
            if_new_data_available = True

        cursor = postgres_connection.cursor()
        cursor.execute(
          "UPDATE constants SET routes_weather=(%s), truck_schedule_data=(%s), traffic_details = (%s), city_weather = (%s)"
          " WHERE id = 1", 
          (routes_weather_df_row,truck_schedule_df_row,traffic_df_row,city_weather_df_row));
        postgres_connection.commit()
        cursor.close()
        postgres_connection.close()
    
    # Save the preprocessed data
    base_dir = "/opt/ml/processing"

    dict_ = {'if_new_data_available': if_new_data_available, 'new_truck_schedule_df_shape':new_truck_schedule_df.shape[0]}
    pd.DataFrame([dict_]).to_csv(f"{base_dir}/result2/result2.csv", header=True, index=False)