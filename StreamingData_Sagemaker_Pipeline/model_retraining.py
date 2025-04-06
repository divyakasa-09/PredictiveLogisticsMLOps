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
    

def split_data_by_date(final_merge, config):
    """
    This function splits the data into training, validation, and test sets based on the given date ranges.
    
    Parameters:
    final_merge (DataFrame): The merged DataFrame.
    config (dict): The configuration dictionary.
    
    Returns:
    tuple: A tuple containing six DataFrames (X_train, y_train, X_valid, y_valid, X_test, y_test).
    """
    try:
        # Specify date ranges
        train_end_date = config['split_date_ranges']['train_end_date']
        test_start_date = config['split_date_ranges']['test_start_date']
    

        # Splitting the data into training, validation, and test sets based on date
        train_df = final_merge[final_merge['estimated_arrival'] <= pd.to_datetime(train_end_date)]
        validation_df = final_merge[
            (final_merge['estimated_arrival'] > pd.to_datetime(train_end_date)) &
            (final_merge['estimated_arrival'] <= pd.to_datetime(test_start_date))
        ]
        test_df = final_merge[final_merge['estimated_arrival'] > pd.to_datetime(test_start_date)]


        return train_df, validation_df, test_df

    except Exception as e:
        # Log the error
        logger.error(f"An error occurred during data splitting: {str(e)}")

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


def save_encoder_scaler(encoder, scaler, encoder_path='output/truck_data_encoder.pkl', scaler_path='output/truck_data_scaler.pkl'):
    try:
        # Dumping the encoder and scaler for future use
        dump(encoder, open(encoder_path, 'wb'))
        dump(scaler, open(scaler_path, 'wb'))
        logger.info("Encoder and scaler saved successfully.")

    except Exception as e:
        # Log the error
        logger.error(f"An error occurred during saving encoder and scaler: {str(e)}")


def processing_data(config, train, valid, test):

    try:
        # Specify
        columns_to_drop_nulls = config['features']['columns_to_drop_null_values']
        encode_columns = config['features']['encode_column_names']
        cts_cols = config['features']['cts_col_names']
        cat_cols = config['features']['cat_col_names']
        target = config['features']['target']

        # Drop null values from specified columns
        train = drop_null_values(train, columns_to_drop_nulls)
        valid = drop_null_values(valid, columns_to_drop_nulls)
        test = drop_null_values(test, columns_to_drop_nulls)

        # Calculate mode on training data
        load_capacity_mode = calculate_mode(train, 'load_capacity_pounds')

        # Fill missing values with mode
        fill_missing_values_with_mode(train, 'load_capacity_pounds', load_capacity_mode)
        fill_missing_values_with_mode(valid, 'load_capacity_pounds', load_capacity_mode)
        fill_missing_values_with_mode(test, 'load_capacity_pounds', load_capacity_mode)

        # One-Hot Encode and Drop original categorical features
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        
        encoder.fit(train[encode_columns])
        train = process_categorical_data(train, encoder, encode_columns)
        valid = process_categorical_data(valid, encoder, encode_columns)
        test = process_categorical_data(test, encoder, encode_columns)

        # Scale data
        scaler = StandardScaler()
        train[cts_cols] = scaler.fit_transform(train[cts_cols])
        valid = scale_data(valid, scaler, cts_cols)
        test = scale_data(test, scaler, cts_cols)

        # Save encoder and scaler
        # save_encoder_scaler(encoder, scaler)

        # Extracting features and target variables
        X_train = train[cts_cols + cat_cols]
        y_train = train[target]

        X_valid = valid[cts_cols + cat_cols]
        y_valid = valid[target]

        X_test = test[cts_cols + cat_cols]
        y_test = test[target]

        # Drop original categorical features
        X_train = X_train.drop(encode_columns, axis=1)
        X_valid = X_valid.drop(encode_columns, axis=1)
        X_test = X_test.drop(encode_columns, axis=1)

        return X_train, y_train, X_valid, y_valid, X_test, y_test

    except Exception as e:
        # Log the error
        logger.error(f"An error occurred while processing the data: {str(e)}")
        
        
def train_logistic_model(X_train, y_train, X_valid, y_valid, X_test, y_test, class_weights, project_config):
    '''
    Train a Logistic Regression model and log performance metrics using Weights & Biases.

    Parameters:
    - X_train, X_valid, X_test: Feature sets for training, validation, and testing.
    - y_train, y_valid, y_test: Target labels for training, validation, and testing.

    Returns:
    - None
    '''
    # Get the column names of features
    features = X_train.columns
    PROJECT_NAME = project_config['wandb']['wandb_project']
    # Initialize Weights & Biases run    
    
    with wandb.init(project=PROJECT_NAME) as run:
        config = wandb.config

        # Define model parameters
        params = {
            "random_state": 13,
            "class_weight": class_weights 
            }

        # Initialize Logistic Regression model
        model = LogisticRegression(**params)

        # Train the model
        model.fit(X_train, y_train)

        # Train predictions and performance
        y_train_pred = model.predict(X_train)
        train_f1_score = f1_score(y_train, y_train_pred)

        # Validation predictions and performance
        y_valid_pred = model.predict(X_valid)
        valid_f1_score = f1_score(y_valid, y_valid_pred)

        # Test predictions and performance
        y_preds = model.predict(X_test)
        y_probas = model.predict_proba(X_test)
        score = f1_score(y_test, y_preds)

        # Log performance metrics
        print(f"F1_score Train: {round(train_f1_score, 4)}")
        print(f"F1_score Valid: {round(valid_f1_score, 4)}")
        print(f"F1_score Test: {round(score, 4)}")

        wandb.log({"f1_score_train": train_f1_score})
        wandb.log({"f1_score_valid": valid_f1_score})
        wandb.log({"f1_score": score})
        

        # Plot classifier performance
        wandb.sklearn.plot_classifier(model, X_train, X_test, y_train, y_test,
                                            y_preds, y_probas, labels= None, model_name='LogisticRegression', feature_names=features)

        # Save the trained model
        model_artifact = wandb.Artifact("LogisticRegression", type="model", metadata=dict(config))
        joblib.dump(model, "log-truck-model.pkl")
        model_artifact.add_file("log-truck-model.pkl")
        wandb.save("log-truck-model.pkl")
        run.log_artifact(model_artifact)



def train_random_forest(X_train, y_train, X_valid, y_valid, X_test, y_test, class_weights, project_config):
    '''
    Train a Random Forest classifier and log performance metrics using Weights & Biases.

    Parameters:
    - X_train, X_valid, X_test: Feature sets for training, validation, and testing.
    - y_train, y_valid, y_test: Target labels for training, validation, and testing.

    Returns:
    - None
    '''
    features = X_train.columns
    labels = ["delay"]
    PROJECT_NAME = project_config['wandb']['wandb_project']

    # Initialize Weights & Biases run
        
    with wandb.init(project=PROJECT_NAME) as run:
        config = wandb.config

        # Define model parameters
        model = RandomForestClassifier(n_estimators=20, class_weight=class_weights, random_state=7)

        # Train the model
        model.fit(X_train, y_train)

        # Train predictions and performance
        y_train_pred = model.predict(X_train)
        train_f1_score = f1_score(y_train, y_train_pred)

        # Validation predictions and performance
        y_valid_pred = model.predict(X_valid)
        valid_f1_score = f1_score(y_valid, y_valid_pred)

        # Test predictions and performance
        y_preds = model.predict(X_test)
        y_probas = model.predict_proba(X_test)
        score = f1_score(y_test, y_preds)

        # Log performance metrics
        print(f"F1_score Train: {round(train_f1_score, 4)}")
        print(f"F1_score Valid: {round(valid_f1_score, 4)}")
        print(f"F1_score Test: {round(score, 4)}")

        wandb.log({"f1_score_train": train_f1_score})
        wandb.log({"f1_score_valid": valid_f1_score})
        wandb.log({"f1_score": score})
        
        # Plot classifier performance
        wandb.sklearn.plot_classifier(model, X_train, X_test, y_train, y_test, y_preds, y_probas, labels=None,
                                      model_name='RandomForestClassifier', feature_names=features)

        # Save the trained model
        model_artifact = wandb.Artifact("RandomForestClassifier", type="model", metadata=dict(config))
        joblib.dump(model, "randomf-truck-model.pkl")
        model_artifact.add_file("randomf-truck-model.pkl")
        wandb.save("randomf-truck-model.pkl")
        run.log_artifact(model_artifact)




def train_xgb_model(X_train, y_train, X_valid, y_valid, X_test, y_test, project_config):
    '''
    Train an XGBoost classifier and log performance metrics using Weights & Biases.

    Parameters:
    - X_train, X_valid, X_test: Feature sets for training, validation, and testing.
    - y_train, y_valid, y_test: Target labels for training, validation, and testing.

    Returns:
    - None
    '''
    PROJECT_NAME = project_config['wandb']['wandb_project']
    features = X_train.columns
    labels = ["delay"]

    # Initialize Weights & Biases run
        
    with wandb.init(project=PROJECT_NAME) as run:
        config = wandb.config

        # Convert training and test sets to DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # Train initial model
        params = {'objective': 'multi:softmax', 'num_class': 2}
        num_rounds = 30
        xgbmodel = xgb.train(params, dtrain, num_rounds, evals=[(dvalid, 'validation')],
                             early_stopping_rounds=10)

        # Train predictions and performance
        y_train_pred = xgbmodel.predict(dtrain)
        train_f1_score = f1_score(y_train, y_train_pred)

        # Validation predictions and performance
        y_valid_pred = xgbmodel.predict(dvalid)
        valid_f1_score = f1_score(y_valid, y_valid_pred)

        # Test predictions and performance
        y_preds = xgbmodel.predict(dtest)
        score = f1_score(y_test, y_preds)

        # Log performance metrics
        print(f"F1_score Train: {round(train_f1_score, 4)}")
        print(f"F1_score Valid: {round(valid_f1_score, 4)}")
        print(f"F1_score Test: {round(score, 4)}")

        wandb.log({"f1_score_train": train_f1_score})
        wandb.log({"f1_score_valid": valid_f1_score})
        wandb.log({"f1_score": score})
        

        # Save the trained model
        model_artifact = wandb.Artifact("XGBoost", type="model", metadata=dict(config))
        joblib.dump(xgbmodel, "xgb-truck-model.pkl")
        model_artifact.add_file("xgb-truck-model.pkl")
        wandb.save("xgb-truck-model.pkl")
        run.log_artifact(model_artifact)






def train_models(X_train, y_train, X_valid, y_valid, X_test, y_test, class_weights, project_config):
    '''
    Train multiple machine learning models and log performance metrics.

    Parameters:
    - X_train, X_valid, X_test: Feature sets for training, validation, and testing.
    - y_train, y_valid, y_test: Target labels for training, validation, and testing.
    - project_config (dict): Configuration parameters for the project.

    Returns:
    - None
    '''
    try:
        # Train Logistic Regression model
        logger.info("Logistic Model Training started.")
        train_logistic_model(X_train, y_train, X_valid, y_valid, X_test, y_test, class_weights, project_config)

        # Train Random Forest model
        logger.info("Random Forest Model Training started.")
        train_random_forest(X_train, y_train, X_valid, y_valid, X_test, y_test, class_weights, project_config)

        # Train XGBoost model
        logger.info("XGB Model Training started.")
        train_xgb_model(X_train, y_train, X_valid, y_valid, X_test, y_test, project_config)

    except Exception as e:
        # Log the error
        logger.error(f"An error occurred while training the models: {str(e)}")

def calculate_class_weights(y_train):
    '''
    Calculate class weights based on the distribution of target labels.
    
    Parameters:
    - y_train: Target labels for training data.
    
    Returns:
    - class_weights (dict): Dictionary containing class weights.
    '''
    class_counts = y_train.value_counts().to_dict()
    
    # Calculate class weights
    weights = len(y_train) / (2 * class_counts[0]), len(y_train) / (2 * class_counts[1])
    
    # Create a dictionary for class weights
    class_weights = {0: weights[0], 1: weights[1]}
    
    return class_weights


def model_retraining(config):
    # Fetch final merge from hopsworks
    final_merge_df = fetch_data(config, 'final_data')
    logger.info("Final merged data fetched from Hopsworks.")
    
    final_merge_df.dropna(subset='delay',inplace=True)

    # Data split
    train_df, validation_df, test_df = split_data_by_date(final_merge_df, config)
    logger.info("Final merged data merged into train, test, validation")

    # Data preprocessing
    X_train, y_train, X_valid, y_valid, X_test, y_test = processing_data(config, train_df, validation_df, test_df)

    class_weights = calculate_class_weights(y_train)

    # Modelling

    logger.info("Model Training started.")
    train_models(X_train, y_train, X_valid, y_valid, X_test, y_test, class_weights, config)
    logger.info("Model Training and Registration Completed")
    
    subject = 'Notification from SageMaker Streaming Pipeline'
    message = 'Model retraining is completed.'
    send_email(subject, message)
    


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
    
    # Model retraining
    model_retraining(config)

    
