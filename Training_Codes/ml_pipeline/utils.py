import logging
import yaml
import hopsworks
import pandas as pd
import os
import wandb
import numpy as np

def setup_logging():
    # defining the logger
    logger=logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    formatter=logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(funcName)s:%(lineno)d:%(message)s')
    
    error_log_path = 'logs/truck_eta_error_logs.log' 
    file_handler_error=logging.FileHandler(os.path.abspath(error_log_path))
    file_handler_error.setFormatter(formatter)
    file_handler_error.setLevel(logging.ERROR)
    logger.addHandler(file_handler_error)
    
    info_log_path = 'logs/truck_eta_info_logs.log' 
    file_handler_info=logging.FileHandler(os.path.abspath(info_log_path))
    file_handler_info.setFormatter(formatter)
    file_handler_info.setLevel(logging.INFO)
    logger.addHandler(file_handler_info)
    
    return logger

# Setup logging
logger = setup_logging()

def load_config(file_path='config.yaml'):
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


def fetch_best_model(config):
    os.environ['WANDB_API_KEY'] = config['wandb']['wandb_api_key']
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

    print(f"Run scores comparison starting")
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

    # logger.info("artifact_dir: %s", artifact_dir)

    return artifact_dir





def drop_null_values(data, columns_to_drop):
    try:
        # Drop rows with null values in specific columns
        processed_data = data.dropna(subset=columns_to_drop).reset_index(drop=True)
        print("Null values dropped successfully.")

        return processed_data

    except Exception as e:
        # Log the error
        print(f"An error occurred during null value removal: {str(e)}")


def fill_missing_values_with_mode(data, column_name, mode_value):
    try:
        # Fill missing values with mode
        data[column_name] = data[column_name].fillna(mode_value)
        print(f"Missing values in {column_name} filled with mode value.")

    except Exception as e:
        # Log the error
        print(f"An error occurred during missing value imputation: {str(e)}")


def process_categorical_data(data, encoder, encode_columns):
    try:
        # One-Hot Encode
        encoded_features = list(encoder.get_feature_names_out(encode_columns))
        data[encoded_features] = encoder.transform(data[encode_columns])
        print("One-Hot Encoding completed successfully.")


        print("Original categorical features dropped successfully.")

        return data

    except Exception as e:
        # Log the error
        print(f"An error occurred during categorical data processing: {str(e)}")


def scale_data(data, scaler, cts_cols):
    try:
        # Scale separate columns using Standard Scaler
        data[cts_cols] = scaler.transform(data[cts_cols])
        print("Data scaling completed successfully.")

        return data

    except Exception as e:
        # Log the error
        print(f"An error occurred during data scaling: {str(e)}")

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
        print(f"An error occurred while processing the data: {str(e)}")
