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
    
    s3_bucket_name = "truck-eta-classification-logs"
    s3_object_key = "reports/"+timestamp+"/"+data_name+"_results.html"
    # Get a pre-signed URL for the uploaded file
    url = s3.generate_presigned_url(
        ClientMethod='get_object',
        Params={'Bucket': s3_bucket_name, 'Key': s3_object_key},
        ExpiresIn=604800  # Adjust the expiration time as needed
    )
    
    url = data_name + "\n" +message + "\nLink to report " + url
    send_email(subject, url)


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
        
def send_email(subject, message):
    sns_client = boto3.client('sns',region_name='us-east-2')
    
    topic_arn = 'arn:aws:sns:us-east-2:143176219551:StreamingNotification'

    response = sns_client.publish(
        TopicArn=topic_arn,
        Subject=subject,
        Message=message
    )
    print(f"Email sent! MessageId: {response['MessageId']}")
    


def fetch_best_model(config):
    PROJECT_NAME = config['wandb']['wandb_project']
    USER_NAME= config['wandb']['wandb_user']
    run = wandb.init(project=PROJECT_NAME)

    # try:
    #     model_artifact = wandb_run.use_artifact(f"{USER_NAME}/{PROJECT_NAME}/XGBoost:latest", type="model")
    #     model_dir = model_artifact.download()
    #     wandb_run.finish()
    #     model_path = os.path.join(model_dir, "xgb-truck-model.pkl")
    #     model = joblib.load(model_dir + "/xgb-truck-model.pkl")
    #     return model
    # except Exception as e:
    #     logger.error(f"An Error occurred while loading the model from wandb: {e}")
        
    runs = wandb.Api().runs(f"{run.entity}/{PROJECT_NAME}")

    # Initialize variables to keep track of the best F1 score and its corresponding run ID
    best_f1_score = -1
    best_run_id = None

    # Iterate through runs to find the best F1 score
    for run in runs:
        # Fetch the metrics logged during the run
        run_metrics = run.history()

        # Check if all three F1 scores are logged in the metrics
        if "f1_score_train" in run_metrics.columns and "f1_score_valid" in run_metrics.columns and "f1_score" in run_metrics.columns:
            # Get the last F1 scores for training, validation, and test sets
            f1_score_train = run_metrics['f1_score_train'].max()
            f1_score_valid = run_metrics['f1_score_valid'].max()
            f1_score_test = run_metrics['f1_score'].max()

            # Calculate the average F1 score to prioritize consistency
            avg_f1_score = np.median([f1_score_train , f1_score_valid ,f1_score_test]) 

            # Update the best F1 score and run ID if a higher F1 score is found
            if avg_f1_score > best_f1_score:
                best_f1_score = avg_f1_score
                best_run_id = run.id

            # print("f1_score_train", f1_score_train)
            # print("f1_score_valid", f1_score_valid)
            # print("f1_score_test", f1_score_test)
            # print("avg_f1_score", avg_f1_score)
            # print("best_f1_score", best_f1_score)

    # Print the best F1 score and its corresponding run ID
    print(f"Best Average F1 Score: {best_f1_score}")
    print(f"Best Run ID: {best_run_id}")

    # Fetch the details of the best run
    best_run = wandb.Api().run(f"{run.entity}/{PROJECT_NAME}/{best_run_id}")

    artifacts = best_run.logged_artifacts()
    best_model = [artifact for artifact in artifacts if artifact.type == 'model'][0]

    artifact_dir = best_model.download()
    
    logger.info("artifact_dir: %s", artifact_dir)
    
    return artifact_dir

def drop_null_values(data, columns_to_drop):
    try:
        # Drop rows with null values in specific columns
        processed_data = data.dropna(subset=columns_to_drop).reset_index(drop=True)
        logger.info("Null values dropped successfully.")

        return processed_data

    except Exception as e:
        # Log the error
        logger.error(f"An error occurred during null value removal: {str(e)}")

def calculate_mode(data, column_name):
    try:
        # Calculate mode on the data
        mode_value = data[column_name].mode().iloc[0]
        logger.info(f"Mode calculated successfully for {column_name}.")

        return mode_value

    except Exception as e:
        # Log the error
        logger.error(f"An error occurred during mode calculation: {str(e)}")


def fill_missing_values_with_mode(data, column_name, mode_value):
    try:
        # Fill missing values with mode
        data[column_name] = data[column_name].fillna(mode_value)
        logger.info(f"Missing values in {column_name} filled with mode value.")

    except Exception as e:
        # Log the error
        logger.error(f"An error occurred during missing value imputation: {str(e)}")


def process_categorical_data(data, encoder, encode_columns):
    try:
        # One-Hot Encode
        encoded_features = list(encoder.get_feature_names_out(encode_columns))
        data[encoded_features] = encoder.transform(data[encode_columns])
        logger.info("One-Hot Encoding completed successfully.")

        
        logger.info("Original categorical features dropped successfully.")

        return data

    except Exception as e:
        # Log the error
        logger.error(f"An error occurred during categorical data processing: {str(e)}")

        

def scale_data(data, scaler, cts_cols):
    try:
        # Scale separate columns using Standard Scaler
        data[cts_cols] = scaler.transform(data[cts_cols])
        logger.info("Data scaling completed successfully.")

        return data

    except Exception as e:
        # Log the error
        logger.error(f"An error occurred during data scaling: {str(e)}")


def processing_new_data(config, data, scaler, encoder):

    try:
        # Specify
        columns_to_drop_nulls = config['features']['columns_to_drop_null_values']
        encode_columns = config['features']['encode_column_names']
        cts_cols = config['features']['cts_col_names']
        cat_cols = config['features']['cat_col_names']
        target = config['features']['target']

        # Drop null values from specified columns
        data = drop_null_values(data, columns_to_drop_nulls)

        # Fill missing values with mode
        fill_missing_values_with_mode(data, 'load_capacity_pounds', 3000)

        # One-Hot Encode and Drop original categorical features
        data = process_categorical_data(data, encoder, encode_columns)

        # Scale data
        data = scale_data(data, scaler, cts_cols)

        # Save encoder and scaler
        # save_encoder_scaler(encoder, scaler)

        # Extracting features and target variables
        X_test = data[cts_cols + cat_cols]
        y_test = data[target]

        X_test = X_test.drop(encode_columns, axis=1)

        return X_test, y_test

    except Exception as e:
        # Log the error
        logger.error(f"An error occurred while processing the data: {str(e)}")


def predict_delay(truck_model, X_data, y_data, merged_data):
    """
    Predict delay using the given truck model and data.

    Parameters:
    - truck_model (xgb.core.Booster or other model): The trained model for predicting delay.
    - X_data (DataFrame): Features used for input
    - y_data (Series): Target variable 
    - merged_data (DataFrame): Merged DataFrame containing truck_id and route_id.

    Returns:
    - DataFrame: DataFrame with delay predictions and additional columns for truck_id, route_id, and actual delay.
    """
    try:
        if isinstance(truck_model, xgb.core.Booster):
            dtrain = xgb.DMatrix(X_data, label=y_data)
            y_pred = truck_model.predict(dtrain)
        else:
            y_pred = truck_model.predict(X_data)

        X_data['Delay_Predictions'] = y_pred
        X_data['truck_id'] = merged_data['truck_id']
        X_data['route_id'] = merged_data['route_id']
  
        X_data['Delay'] = y_data
    except Exception as e:
        print(f"An error occurred during prediction {e}")

    return X_data


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
    
    subject = 'Notification from SageMaker Streaming Pipeline'
    message = 'Data drift is not detected. \nChecking for Model drift.'
    send_email(subject, message)

    # Loading encoder and scaler from S3
    pathlib.Path(f"{base_dir}/output").mkdir(parents=True, exist_ok=True)

    fn = f"{base_dir}/output/truck_data_encoder.pkl"

    s3 = boto3.resource("s3")
    s3.Bucket("my-artifacts1").download_file("truck_data_encoder.pkl", fn)

    truck_data_encoder = joblib.load(fn)

    logger.info(truck_data_encoder)

    fn = f"{base_dir}/output/truck_data_scaler.pkl"

    print(fn)

    s3 = boto3.resource("s3")
    s3.Bucket("my-artifacts1").download_file("truck_data_scaler.pkl", fn)

    truck_data_scaler = joblib.load(fn)

    logger.info(truck_data_encoder)


    # Fetching best model from wandb

    model_dir = fetch_best_model(config)

    dir_list = os.listdir(model_dir)

    pkl_files = [file for file in dir_list if file.endswith(".pkl")]

    if pkl_files:
        # Assuming there is at least one .pkl file, load the first one
        pkl_file_path = os.path.join(model_dir, pkl_files[0])
        truck_model = joblib.load(pkl_file_path)

    logger.info("model loaded")

    logger.info(truck_model)

    logger.info("preprocessing new data")
    
    postgres_connection = create_postgres_connection(config)

    logger.info("Created connection to Postgres Databases")

    # Fetch data from PostgreSQL tables
    constants = read_data(postgres_connection, "constants")
    
    min_start_number_of_week = constants.at[0,'week_start_date']

    lower_date = (min_start_number_of_week) - timedelta(days=7)
    upper_date = (min_start_number_of_week) - timedelta(days=1)

    lower_date = str(lower_date) + " 00:00:00"
    upper_date = str(upper_date)+ " 23:59:59"
    
    print(lower_date)
    print(upper_date)

    final_merge_df = fetch_data(config,'final_data')

    final_merge_df = final_merge_df.dropna(subset=['delay']).reset_index(drop=True)

    final_merge_old = final_merge_df[(final_merge_df['estimated_arrival']<lower_date)]
    final_merge_new = final_merge_df[(final_merge_df['estimated_arrival']>=lower_date) & (final_merge_df['estimated_arrival']<=upper_date)]

    # Reference Data Predictions
    X_train, y_train = processing_new_data(config, final_merge_old, truck_data_scaler, truck_data_encoder)
    reference = predict_delay(truck_model, X_train, y_train, final_merge_old)


    # Current Data Predictions
    X_test, y_test = processing_new_data(config, final_merge_new, truck_data_scaler, truck_data_encoder)
    current = predict_delay(truck_model, X_test, y_test, final_merge_new)


    # Model Drift report
    from evidently.metric_preset import ClassificationPreset
    classification_performance_report = Report(metrics=[ ClassificationPreset()])

    current['truck_id'] = final_merge_new['truck_id']
    current['route_id'] = final_merge_new['route_id']

    current['Delay'] = current['Delay'].astype('int32')
    current['Delay_Predictions'] = current['Delay_Predictions'].astype('int32')
    reference['Delay'] = reference['Delay'].astype('int32')
    reference['Delay_Predictions'] = reference['Delay_Predictions'].astype('int32')
    
    unnecessary_cols = ['unique_id', 'truck_id', 'route_id', 'departure_date',
               'estimated_arrival', 'route_description',
               'estimated_arrival_nearest_hour', 'departure_date_nearest_hour',
               'origin_id', 'destination_id', 'origin_description',
               'destination_description',  'driver_id', 'name', 'gender', 'fuel_type',
               'driving_style',  'vehicle_no']


    target = 'Delay'
    prediction = 'Delay_Predictions'
    numerical_features =  ['route_avg_temp', 'route_avg_wind_speed',
            'route_avg_precip', 'route_avg_humidity', 'route_avg_visibility',
            'route_avg_pressure', 'distance', 'average_hours',
            'origin_temp', 'origin_wind_speed', 'origin_precip', 'origin_humidity',
             'origin_pressure',
            'destination_temp','destination_wind_speed','destination_precip',
            'destination_humidity', 'destination_pressure',
              'avg_no_of_vehicles', 'truck_age','load_capacity_pounds', 'mileage_mpg',
              'age', 'experience','average_speed_mph']
    categorical_features = ['origin_visibility','destination_visibility', 'accident',  'ratings','is_midnight']


    # 8. Data Drift check
    column_mapping = ColumnMapping()

    column_mapping.target = target
    column_mapping.prediction = prediction
    column_mapping.numerical_features = numerical_features
    column_mapping.categorical_features = categorical_features


    classification_performance_report.run(reference_data=reference, current_data=current, column_mapping = column_mapping)


    subject = 'Notification from SageMaker Streaming Pipeline'
    message = classification_performance_report.as_dict()
    #send_email(subject, message)
    s3 = boto3.client('s3')
    classification_performance_report.save_html(base_dir+"model_drift_results.html")

    s3.upload_file(base_dir+"model_drift_results.html", "truck-eta-classification-logs", "reports/"+timestamp+"/model_drift_results.html")
    
    evaluation_results = {}
    
    evaluation_results['f1_score'] = message['metrics'][0]['result']['current']['f1']   
    
    if message['metrics'][0]['result']['current']['f1'] < 0.6:
        subject = 'Notification from SageMaker Streaming Pipeline'
        message = 'Model drift detected, F1 score is '+ str(message['metrics'][0]['result']['current']['f1'])+'. Model retraining pipeline is initiated.'
        send_email(subject, message)

    output_dir = "/opt/ml/processing/evaluation3"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation3.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(evaluation_results))      