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
import pandas as pd
from evidently.test_suite import TestSuite
from evidently.tests import *
from datetime import datetime
import time
from evidently.metrics import *



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
    
    

def send_new_data_to_hopsworks(new_routes_weather_df, new_truck_schedule_df, new_traffic_details_df, new_city_weather_df, config):
    """
    Send new data to Hopsworks feature groups.

    This function takes in four dataframes containing new data for different feature groups,
    calculates the total number of new rows, and sends the data to Hopsworks if there are any new rows.

    Parameters:
    new_routes_weather_df (pandas.DataFrame): DataFrame containing new data for the route_weather_details_fg feature group.
    new_truck_schedule_df (pandas.DataFrame): DataFrame containing new data for the truck_schedule_details_fg feature group.
    new_traffic_details_df (pandas.DataFrame): DataFrame containing new data for the traffic_details_fg feature group.
    new_city_weather_df (pandas.DataFrame): DataFrame containing new data for the city_weather_details_fg feature group.
    config (dict): Configuration dictionary containing Hopsworks API key.

    """
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

        logger.info("All done")


def processing_and_feature_engg(routes_df, route_weather, drivers_df, trucks_df, traffic_df, schedule_df, weather_df, start_number):
    
    drivers_df=drivers_df.drop(columns=['event_time'])

    trucks_df=trucks_df.drop(columns=['event_time'])

    routes_df=routes_df.drop(columns=['event_time'])

    # drop duplicates
    weather_df=weather_df.drop_duplicates(subset=['city_id','date','hour'])

    # drop unnecessary cols
    weather_df=weather_df.drop(columns=['chanceofrain','chanceoffog','chanceofsnow','chanceofthunder'])

    # Convert 'hour' to a 4-digit string format
    #weather_df['hour'] = weather_df['hour'].astype(int)
    weather_df['hour'] = weather_df['hour'].apply(lambda x: f'{x:04d}')

    # Convert 'hour' to datetime format
    weather_df['hour'] = pd.to_datetime(weather_df['hour'], format='%H%M').dt.time

    # Combine 'date' and 'hour' to create a new datetime column 'custom_date' and insert it at index 1
    weather_date_val = pd.to_datetime(weather_df['date'].astype(str) + ' ' + weather_df['hour'].astype(str))
    weather_df.insert(1, 'custom_date', weather_date_val)

    # Drop unnecessary cols
    route_weather=route_weather.drop(columns=['chanceofrain','chanceoffog','chanceofsnow','chanceofthunder'])

    traffic_df=traffic_df.drop_duplicates(subset=['route_id','date','hour'],keep='first')

    # Convert 'hour' to a 4-digit string format
    #traffic_df['hour'] = traffic_df['hour'].astype(int)
    traffic_df['hour'] = traffic_df['hour'].apply(lambda x: f'{x:04d}')

    # Convert 'hour' to datetime format
    traffic_df['hour'] = pd.to_datetime(traffic_df['hour'], format='%H%M').dt.time

    # Combine 'date' and 'hour' to create a new datetime column 'custom_date' and insert it at index 1
    traffic_custom_date = pd.to_datetime(traffic_df['date'].astype(str) + ' ' + traffic_df['hour'].astype(str))
    traffic_df.insert(1, 'custom_date', traffic_custom_date)

    # Feature Engineering
    ## Insert a new column 'unique_id' with unique identifiers starting from the specified number, and return the min max off this for reference and fetching from hopsworks
    schedule_df.insert(0, 'unique_id', np.arange(start_number, start_number + len(schedule_df), dtype=np.int64))
    #schedule_df.insert(0, 'unique_id', np.arange(len(schedule_df), dtype=np.int64))
    #schedule_df.insert(0, 'unique_id', np.arange(len(schedule_df), dtype=np.int64))
    nearest_6h_schedule_df=schedule_df.copy()
    nearest_6h_schedule_df['estimated_arrival']=nearest_6h_schedule_df['estimated_arrival'].dt.ceil("6H")
    nearest_6h_schedule_df['departure_date']=nearest_6h_schedule_df['departure_date'].dt.floor("6H")


    # Assign a new column 'date' using a list comprehension to generate date ranges between 'departure_date' and 'estimated_arrival' with a frequency of 6 hours
    # This will create a list of date ranges for each row
    # Explode the 'date' column to create separate rows for each date range

    exploded_6h_scheduled_df=(nearest_6h_schedule_df.assign(date = [pd.date_range(start, end, freq='6H')
                        for start, end
                        in zip(nearest_6h_schedule_df['departure_date'], nearest_6h_schedule_df['estimated_arrival'])]).explode('date', ignore_index = True))


    schduled_weather=exploded_6h_scheduled_df.merge(route_weather,on=['route_id','date'],how='left')

    # Define a custom function to calculate mode
    def custom_mode(x):
        return x.mode().iloc[0]

    # Group by specified columns and aggregate
    schedule_weather_grp = schduled_weather.groupby(['unique_id','truck_id','route_id'], as_index=False).agg(
        route_avg_temp=('temp','mean'),
        route_avg_wind_speed=('wind_speed','mean'),
        route_avg_precip=('precip','mean'),
        route_avg_humidity=('humidity','mean'),
        route_avg_visibility=('visibility','mean'),
        route_avg_pressure=('pressure','mean'),
        route_description=('description', custom_mode)
    )

    schedule_weather_merge=schedule_df.merge(schedule_weather_grp,on=['unique_id','truck_id','route_id'],how='left')

    #take hourly as weather data available hourly
    nearest_hour_schedule_df=schedule_df.copy()
    nearest_hour_schedule_df['estimated_arrival_nearest_hour']=nearest_hour_schedule_df['estimated_arrival'].dt.round("H")
    nearest_hour_schedule_df['departure_date_nearest_hour']=nearest_hour_schedule_df['departure_date'].dt.round("H")
    nearest_hour_schedule_route_df=pd.merge(nearest_hour_schedule_df, routes_df, on='route_id', how='left')


    # Create a copy of the 'weather_df' DataFrame for manipulation
    origin_weather_data = weather_df.copy()

    # Drop the 'date' and 'hour' columns from 'origin_weather_data'
    origin_weather_data = origin_weather_data.drop(columns=['date', 'hour'])

    origin_weather_data.columns = ['origin_id','departure_date_nearest_hour', 'origin_temp', 'origin_wind_speed','origin_description', 'origin_precip',
        'origin_humidity', 'origin_visibility', 'origin_pressure']

    # Create a copy of the 'weather_df' DataFrame for manipulation
    destination_weather_data = weather_df.copy()

    # Drop the 'date' and 'hour' columns from 'destination_weather_data'
    destination_weather_data = destination_weather_data.drop(columns=['date', 'hour'])

    destination_weather_data.columns = ['destination_id', 'estimated_arrival_nearest_hour','destination_temp', 'destination_wind_speed','destination_description', 'destination_precip',
        'destination_humidity', 'destination_visibility', 'destination_pressure' ]

    # Merge 'nearest_hour_schedule_route_df' with 'origin_weather_data' based on specified columns
    origin_weather_merge = pd.merge(nearest_hour_schedule_route_df, origin_weather_data, on=['origin_id','departure_date_nearest_hour'], how='left')

    # Merge 'origin_weather_merge' with 'destination_weather_data' based on specified columns
    origin_destination_weather = pd.merge(origin_weather_merge, destination_weather_data , on=['destination_id', 'estimated_arrival_nearest_hour'], how='left')

    # Create a copy of the schedule DataFrame for manipulation
    nearest_hour_schedule_df = schedule_df.copy()

    # Round 'estimated_arrival' times to the nearest hour
    nearest_hour_schedule_df['estimated_arrival'] = nearest_hour_schedule_df['estimated_arrival'].dt.round("H")

    # Round 'departure_date' times to the nearest hour
    nearest_hour_schedule_df['departure_date'] = nearest_hour_schedule_df['departure_date'].dt.round("H")

    hourly_exploded_scheduled_df=(nearest_hour_schedule_df.assign(custom_date = [pd.date_range(start, end, freq='H')  # Create custom date ranges
                        for start, end
                        in zip(nearest_hour_schedule_df['departure_date'], nearest_hour_schedule_df['estimated_arrival'])])  # Using departure and estimated arrival times
                        .explode('custom_date', ignore_index = True))  # Explode the DataFrame based on the custom date range

    scheduled_traffic=hourly_exploded_scheduled_df.merge(traffic_df,on=['route_id','custom_date'],how='left')

    # Define a custom aggregation function for accidents
    def custom_agg(values):
        """
        Custom aggregation function to determine if any value in a group is 1 (indicating an accident).

        Args:
        values (iterable): Iterable of values in a group.

        Returns:
        int: 1 if any value is 1, else 0.
        """
        if any(values == 1):
            return 1
        else:
            return 0

    # Group by 'unique_id', 'truck_id', and 'route_id', and apply custom aggregation
    scheduled_route_traffic = scheduled_traffic.groupby(['unique_id', 'truck_id', 'route_id'], as_index=False).agg(
        avg_no_of_vehicles=('no_of_vehicles', 'mean'),
        accident=('accident', custom_agg)
    )

    origin_destination_weather_traffic_merge=origin_destination_weather.merge(scheduled_route_traffic,on=['unique_id','truck_id','route_id'],how='left')

    merged_data_weather_traffic=pd.merge(schedule_weather_merge, origin_destination_weather_traffic_merge, on=['unique_id', 'truck_id', 'route_id', 'departure_date',
        'estimated_arrival', 'delay'], how='left')

    merged_data_weather_traffic_trucks = pd.merge(merged_data_weather_traffic, trucks_df, on='truck_id', how='left')

    # Merge merged_data with truck_data based on 'truck_id' column (Left Join)
    final_merge = pd.merge(merged_data_weather_traffic_trucks, drivers_df, left_on='truck_id', right_on = 'vehicle_no', how='left')

    # Function to check if there is nighttime involved between arrival and departure time
    def has_midnight(start, end):
        return int(start.date() != end.date())

    try:
        # Apply the function to create a new column indicating nighttime involvement
        final_merge['is_midnight']= final_merge.apply(lambda row: has_midnight(row['departure_date'], row['estimated_arrival']), axis=1)
    except Exception as e:
        print(e)
    fs_data = final_merge.sort_values(["estimated_arrival","unique_id"])
    fs_data['origin_description'] = fs_data['origin_description'].fillna("Unknown")

    return fs_data


def fix_datasets_final_merge(dataframe, result_json):
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
                fixed_df = fixed_df.dropna(subset=list(fixed_df.drop('delay', axis=1).columns)).reset_index(drop=True)

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
                    if column_name != "delay":
                        if expected_type == 'object_':
                            fixed_df[column_name] = fixed_df[column_name].astype('object')
                        else:
                        # Convert the column to the expected type
                            fixed_df[column_name] = fixed_df[column_name].astype(expected_type)

    return fixed_df



if __name__ == "__main__":
    
    import argparse

    # Initialize the ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add an argument for the config file data path
    parser.add_argument("--config-file-data", type=str, required=True, help="Path to the config data")

    # Parse the arguments
    args = parser.parse_args()

    # Initialize the logger
    logger.debug("Starting fetching data")

    # Define the base directory
    base_dir = "/opt/ml/processing"

    # Create the data directory if it doesn't exist
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)

    # Get the config file data path
    config_file = args.config_file_data

    # Extract the bucket and key from the config file data path
    bucket = config_file.split("/")[2]
    key = "/".join(config_file.split("/")[3:])

    # Print the bucket and key for debugging
    print(bucket, key)

    # Log the data source
    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)

    # Define the local file path
    fn = f"{base_dir}/data/config.yaml"

    # Print the local file path
    print(fn)

    # Initialize the S3 resource
    s3 = boto3.resource("s3")

    # Download the file from S3 to the local file path
    s3.Bucket(bucket).download_file(key, fn)

    # Print the local file path after downloading
    print(fn)

    # Load the configuration from the local file path
    config = load_config(fn)

    # Remove the local file
    os.unlink(fn)

    # Set the wandb API key from the configuration
    os.environ['WANDB_API_KEY'] = config['wandb']['wandb_api_key']

    # Get the current timestamp as a formatted string
    timestamp = datetime.now().strftime("%m-%d-%Y")

    # Define the result file path
    result = "/opt/ml/processing/result2/result2.csv"

    # Load the result file as a pandas DataFrame
    result = pd.read_csv(result)
    


    # Check if new data is available and if there are new entries in the truck schedule dataframe
    if ((result.at[0,'if_new_data_available']==True) and (result.at[0,'new_truck_schedule_df_shape']!=0)):
        
        # Log that new data is available and that the program will wait for it to be updated
        logger.info("New data available.")
        logger.info("Waiting for new data to be updated in hopsworks!")
        

        # Pause the program for 900 seconds (15 minutes) to allow for data to be updated
        time.sleep(900)
        logger.info("New data has been updated in hopsworks!")
        
 
         # Fetch all required dataframes from hopsworks 
        routes_weather_df = fetch_data(config, 'route_weather_details_fg')
        truck_schedule_df = fetch_data(config, 'truck_schedule_details_fg')
        traffic_df = fetch_data(config, 'traffic_details_fg')
        city_weather_df = fetch_data(config, 'city_weather_details_fg')
        routes_df = fetch_data(config,'routes_details_fg')
        drivers_df = fetch_data(config,'drivers_details_fg')
        trucks_df = fetch_data(config,'truck_details_fg')
        final_merge_df = fetch_data(config,'final_data')
    
        # Selecting rows from truck_schedule_df which are not processed
        # These rows are present in truck schedule dataframe but not in final merge dataframe
        df_temp = truck_schedule_df.merge(
        final_merge_df[['truck_id','route_id','departure_date']], 
        on=['truck_id','route_id','departure_date'], 
        how='left', 
        indicator=True
        )

        # Creating a new dataframe with the left_only rows
        new_truck_schedule_df = df_temp[df_temp['_merge']=='left_only'].reset_index(drop=True).drop('_merge',axis=1)

        # Filtering the rows based on estimated arrival time
        # Only keeping rows where estimated arrival is less than or equal to the latest date in city_weather_df
        filtering_criteria = new_truck_schedule_df['estimated_arrival'] <= pd.to_datetime(str(city_weather_df['date'].max()) + " 23:59:59")
        new_truck_schedule_df_filtered = new_truck_schedule_df[filtering_criteria]

        # Calculating the number of rows that were not processed
        failed_no_of_rows = new_truck_schedule_df.shape[0] - new_truck_schedule_df_filtered.shape[0]

        # Preparing the subject and message for the notification email
        subject = 'Notification from SageMaker Streaming Pipeline'
        if failed_no_of_rows == 0:
            message = f"All {new_truck_schedule_df.shape[0]} rows have been successfully processed."
        else:
            message = f"Attention: {failed_no_of_rows} row(s) require further action. These rows were not processed successfully because we are still awaiting weather data for the corresponding date(s)."

        # Sending the notification email
        send_email(subject, message)

    
    


        # Initialize the start number for the unique_id by finding the maximum value
        # in the 'unique_id' column of the final_merge_df dataframe
        start_number = final_merge_df['unique_id'].max() + 1

        # Call the processing_and_feature_engg function with the necessary dataframes
        # and save the result in the final_merge_new dataframe
        final_merge_new = processing_and_feature_engg(
            routes_df,
            routes_weather_df,
            drivers_df,
            trucks_df,
            traffic_df,
            new_truck_schedule_df_filtered,
            city_weather_df,
            start_number
        )

        # Fetch the final data from the specified location
        final_merge = fetch_data(config, 'final_data')

        # Run data quality checks on the final merge data and store the results
        # in the final_merge_results variable
        final_merge_dashboard, final_merge_results = run_data_quality_check(
            'Final_merge',
            final_merge.drop('delay', axis=1),
            final_merge_new.drop('delay', axis=1),
            timestamp
        )

        final_merge_new = fix_datasets_final_merge(final_merge_new, final_merge_results)

        final_merge_new['delay'] = final_merge_new['delay'].astype('Int64')


        if final_merge_new.shape[0]>0:
            print("In")
            final_merge_fg = fs.get_feature_group('final_data', version=1)
            final_merge_fg.insert(final_merge_new)
            
            subject = 'Notification from SageMaker Streaming Pipeline'
            message = 'Total number of '+ str(final_merge_new.shape[0]) + ' rows added in final_data feature group in hopsworks.'
            send_email(subject, message)
            
        subject = 'Notification from SageMaker Streaming Pipeline'
        message = 'Streaming Data Pipeline successfully executed. New data is now available in the final data feature group for predictions.!!!!'
        send_email(subject, message)


    else:
        logger.info("No data available for truck schedule table")
        subject = 'Notification from SageMaker Streaming Pipeline'
        message = 'No trucks scheduled for departure today.'
        send_email(subject, message)

    
    # Save the preprocessed data
    base_dir = "/opt/ml/processing"

    dict_ = {'Result': 'Done'}
    pd.DataFrame([dict_]).to_csv(f"{base_dir}/result3/result3.csv", header=True, index=False)