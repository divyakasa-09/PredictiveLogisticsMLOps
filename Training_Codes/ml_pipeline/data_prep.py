# Import the necessary libraries
import psycopg2
import pymysql
import pandas as pd
import logging
import hopsworks
import numpy as np
import yaml
import os
# Setup logging


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

def pre_process_datasets(routes_df, route_weather, drivers_df, trucks_df, traffic_df, schedule_df, weather_df, config):
    '''
    Perform pre-processing on the input datasets.

    Parameters:
    - routes_df (pd.DataFrame): DataFrame for routes data.
    - route_weather (pd.DataFrame): DataFrame for route weather data.
    - drivers_df (pd.DataFrame): DataFrame for drivers data.
    - trucks_df (pd.DataFrame): DataFrame for trucks data.
    - traffic_df (pd.DataFrame): DataFrame for traffic data.
    - schedule_df (pd.DataFrame): DataFrame for truck schedule data.
    - weather_df (pd.DataFrame): DataFrame for city weather data.

    Returns:
    - (pd.DataFrame): Processed DataFrames for each input dataset.
    '''
    try:

        # Rename columns to lowercase
        route_weather = route_weather.rename(columns={'Date': 'date'})
        
        # Convert date columns to datetime
        weather_df['date'] = pd.to_datetime(weather_df['date'])
        route_weather['date'] = pd.to_datetime(route_weather['date'])
        traffic_df['date'] = pd.to_datetime(traffic_df['date'])
        schedule_df['departure_date'] = pd.to_datetime(schedule_df['departure_date'])
        schedule_df['estimated_arrival'] = pd.to_datetime(schedule_df['estimated_arrival'])
        route_weather['date'] = pd.to_datetime(route_weather['date'])

        drivers_df['event_time'] = pd.to_datetime(config['hopsworks']['event_time'])

        # Filling null values with 'Unknown'
        drivers_df['driving_style'] = drivers_df['driving_style'].fillna('Unknown')
        drivers_df['gender'] = drivers_df['gender'].fillna('Unknown')

        trucks_df['fuel_type'] = trucks_df['fuel_type'].replace("", 'Unknown')
        trucks_df['event_time'] = pd.to_datetime(config['hopsworks']['event_time'])

        routes_df['event_time'] = pd.to_datetime(config['hopsworks']['event_time'])

        # Sorting DataFrames
        drivers_df=drivers_df.sort_values(["event_time","driver_id"])
        trucks_df = trucks_df.sort_values(["event_time", "truck_id"])
        routes_df = routes_df.sort_values(["event_time", "route_id"])
        schedule_df = schedule_df.sort_values(["estimated_arrival", "truck_id"])
        traffic_df = traffic_df.sort_values(['date', 'route_id', 'hour'])
        weather_df = weather_df.sort_values(['date', 'city_id', 'hour'])
        route_weather = route_weather.sort_values(by=['date', 'route_id'])
    
    except Exception as e:
        print("Error occured while preprocessing raw datasets before storing them in features stores:", str(e))

    else:
        return routes_df, route_weather, drivers_df, trucks_df, traffic_df, schedule_df, weather_df




def create_feature_group(fs, name, version, description, primary_key, event_time, online_enabled):
    return fs.get_or_create_feature_group(
        name=name,
        version=version,
        description=description,
        primary_key=primary_key,
        event_time=event_time,
        online_enabled=online_enabled
    )

def update_feature_descriptions(fg, feature_descriptions):
    for desc in feature_descriptions:
        fg.update_feature_description(desc["name"], desc["description"])

def configure_statistics(fg, enabled, histograms, correlations):
    fg.statistics_config = {
        "enabled": enabled,
        "histograms": histograms,
        "correlations": correlations
    }

def update_statistics_config(fg):
    fg.update_statistics_config()

def compute_statistics(fg):
    fg.compute_statistics()

def create_and_update_feature_groups(fs, df, name, version, description, primary_key, event_time, online_enabled, feature_descriptions, enabled=True, histograms=True, correlations=True):
    fg = create_feature_group(fs, name, version, description, primary_key, event_time, online_enabled)
    fg.insert(df)
    # Update feature descriptions for drivers
    update_feature_descriptions(fg, feature_descriptions)

    # Configure statistics for the drivers feature group
    configure_statistics(fg, enabled, histograms, correlations)

    # Update the statistics configuration for the drivers feature group
    update_statistics_config(fg)

    # Compute statistics for the drivers feature group
    compute_statistics(fg)




def processing_and_feature_engg(routes_df, route_weather, drivers_df, trucks_df, traffic_df, schedule_df, weather_df, start_number):
    try:
        drivers_df=drivers_df.drop(columns=['event_time'])

        trucks_df=trucks_df.drop(columns=['event_time'])

        routes_df=routes_df.drop(columns=['event_time'])

        # drop duplicates
        weather_df=weather_df.drop_duplicates(subset=['city_id','date','hour'])

        # drop unnecessary cols
        weather_df=weather_df.drop(columns=['chanceofrain','chanceoffog','chanceofsnow','chanceofthunder'])

        # Convert 'hour' to a 4-digit string format
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
        traffic_df['hour'] = traffic_df['hour'].apply(lambda x: f'{x:04d}')

        # Convert 'hour' to datetime format
        traffic_df['hour'] = pd.to_datetime(traffic_df['hour'], format='%H%M').dt.time

        # Combine 'date' and 'hour' to create a new datetime column 'custom_date' and insert it at index 1
        traffic_custom_date = pd.to_datetime(traffic_df['date'].astype(str) + ' ' + traffic_df['hour'].astype(str))
        traffic_df.insert(1, 'custom_date', traffic_custom_date)

        # Feature Engineering
        schedule_df.insert(0, 'unique_id', np.arange(start_number, start_number + len(schedule_df), dtype=np.int64))
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


        # Apply the function to create a new column indicating nighttime involvement
        final_merge['is_midnight'] = final_merge.apply(lambda row: has_midnight(row['departure_date'], row['estimated_arrival']), axis=1)

        fs_data = final_merge.sort_values(["estimated_arrival","unique_id"])
        fs_data['origin_description'] = fs_data['origin_description'].fillna("Unknown")


    except Exception as e:
        print("Error while creating the final merged data:", str(e))

    else:
        return fs_data



def data_reading_and_feature_str_creation(config, postgres_connection, mysql_connection):

        try:

            # Fetch data from PostgreSQL tables
            routes_df = read_data(postgres_connection, config['table_names']['routes_data'])
            route_weather = read_data(postgres_connection, config['table_names']['route_weather'])

            # Fetch data from MySQL tables
            drivers_df = read_data(mysql_connection, config['table_names']['drivers_data'])
            trucks_df = read_data(mysql_connection, config['table_names']['trucks_data'])
            traffic_df = read_data(mysql_connection, config['table_names']['traffic_data'])
            schedule_df = read_data(mysql_connection, config['table_names']['schedule_data'])
            weather_df = read_data(mysql_connection, config['table_names']['weather_data'])

            routes_df, route_weather, drivers_df, trucks_df, traffic_df, schedule_df, weather_df = pre_process_datasets(routes_df, route_weather, 
                                                                                                drivers_df, trucks_df, traffic_df, 
                                                                                                schedule_df, weather_df, config)


            # Login to hopsworks by entering API key value
            project = hopsworks.login(api_key_value=config['hopsworks']['api_key'])

            # Get the feature store
            fs = project.get_feature_store()

            create_and_update_feature_groups(fs, drivers_df, "drivers_details_fg",
                                            1, "Drivers data", ["driver_id"],
                                            "event_time", False, config['data_descriptions']["drivers_details"])
            
            create_and_update_feature_groups(fs, trucks_df, "truck_details_fg",
                                            1, "Truck data", ["truck_id"],
                                            "event_time", False, config['data_descriptions']["truck_details"])
            
            create_and_update_feature_groups(fs, routes_df, "routes_details_fg",
                                            1, "Routes data", ["route_id"],
                                            "event_time", False, config['data_descriptions']["routes_details"])
            
            create_and_update_feature_groups(fs, schedule_df, "truck_schedule_details_fg",
                                            1, "Truck Schedule data", ['truck_id','route_id'],
                                            "estimated_arrival", True, config['data_descriptions']["schedule_details"])
            
            create_and_update_feature_groups(fs, traffic_df, "traffic_details_fg",
                                            1, "Traffic data", ['route_id','hour'],
                                            "date", True, config['data_descriptions']["traffic_details"])
            

            create_and_update_feature_groups(fs, weather_df, "city_weather_details_fg",
                                            1, "City Weather data", ['city_id','hour'],
                                            "date", True, config['data_descriptions']["city_weather_details"])
            
            create_and_update_feature_groups(fs, route_weather, "route_weather_details_fg",
                                            1, "Route Weather data", ['route_id'],
                                            "date", True, config['data_descriptions']["route_weather_details"])

            final_merge = processing_and_feature_engg(routes_df, route_weather, drivers_df, trucks_df, traffic_df, schedule_df, weather_df)

            # final_merge.to_csv("final_merge.csv",index=False)
            create_and_update_feature_groups(fs, final_merge, "final_data",
                                            1, "Truck ETA Final Data", ['unique_id'],
                                            "estimated_arrival", True, config['data_descriptions']["final_data_details"])
        except Exception as e:  
            print(str(e))

def main():
    # Load configuration from config.yaml

    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    postgres_connection = create_postgres_connection(config)
    mysql_connection = create_mysql_connection(config)
    data_reading_and_feature_str_creation(config, postgres_connection, mysql_connection)

if __name__ == '__main__':
    main()

