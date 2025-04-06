import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pickle import dump
import logging
from .utils import setup_logging

# Setup logging
logger = setup_logging()

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
        save_encoder_scaler(encoder, scaler)

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
