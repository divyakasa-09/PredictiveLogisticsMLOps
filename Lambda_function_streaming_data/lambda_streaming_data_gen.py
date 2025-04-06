import json
import pandas as pd
from sqlalchemy import create_engine
from datetime import timedelta
import psycopg2
import boto3
import pymysql

def lambda_handler(event, context):
    # TODO implement
    
    # Your existing code for database connection
    connection = psycopg2.connect(
        user="",
        password="",
        host="",
        database="",
        port=""
    )

    weekly_streaming_df = pd.read_sql('select * from constants;', connection)

    print('Today date ',weekly_streaming_df['day'][0])
    
    s3 = boto3.client('s3')

    # Specify your S3 bucket and file name
    s3_bucket = 'truck-eta-classification'
    s3_prefix = 'streaming_data/'
    
    streaming_routes_weather = s3_prefix + 'streaming_route_weather.csv'  
    local_filename = '/tmp/streaming_route_weather.csv'
    s3.download_file(s3_bucket, streaming_routes_weather, local_filename)
    streaming_routes_weather = pd.read_csv(local_filename)

    streaming_city_weather = s3_prefix + 'streaming_city_weather.csv'  
    local_filename = '/tmp/streaming_city_weather.csv'
    s3.download_file(s3_bucket, streaming_city_weather, local_filename)
    streaming_city_weather = pd.read_csv(local_filename)


    streaming_traffic = s3_prefix + 'streaming_traffic.csv'  
    local_filename = '/tmp/streaming_traffic.csv'
    s3.download_file(s3_bucket, streaming_traffic, local_filename)
    streaming_traffic = pd.read_csv(local_filename)
    
    
    streaming_schedule = s3_prefix + 'streaming_schedule.csv'  
    local_filename = '/tmp/streaming_schedule.csv'
    s3.download_file(s3_bucket, streaming_schedule, local_filename)
    streaming_schedule = pd.read_csv(local_filename)

    
    # Extract the current date
    max_date = weekly_streaming_df['day']
    not_15 = False

    
    # Check if the maximum date is on '2019-02-15' without the specific time
    if str(max_date[0]).split(" ")[0] == '2019-02-15':
        print("if")
        # If yes, set up a time range from 6 AM to 6 PM of that day
        lower_date = str(max_date[0]).split(" ")[0] + " 06:00:00"
        upper_date = str(max_date[0]).split(" ")[0] + " 18:00:00"
        # Record the search date
        search_date = str(max_date[0]).split(" ")[0]
        
        # If the maximum date doesn't match the above conditions
    else:
        print("else")
        not_15 = True
        # Set up a time range from midnight to 6 PM of the next day
        lower_date = str(max_date[0]).split(" ")[0] + " 00:00:00"
        upper_date = str(max_date[0]).split(" ")[0] + " 18:00:00"
        # Record the search date
        search_date = str(max_date[0]).split(" ")[0]
        
        
    streaming_schedule_df = streaming_schedule[(streaming_schedule['departure_date']>=lower_date) & (streaming_schedule['departure_date']<=upper_date)]
        
    first_day_of_the_week = False
    if weekly_streaming_df['weekly_streaming'][0]==1:
        
        print("First day of the week: ")
        first_day_of_the_week = True
        lower_date_first_day = str(max_date[0]).split(" ")[0] + " 00:00:00"
        upper_date_seventh_day = str((max_date[0]) + timedelta(days=6)) + " 23:59:59"
        
        print("lower_date_first_day ",lower_date_first_day)
        print("upper_date_seventh_day ",upper_date_seventh_day)
        
        streaming_routes_weather_df = streaming_routes_weather[(streaming_routes_weather['Date']>=lower_date_first_day) & (streaming_routes_weather['Date']<=upper_date_seventh_day)]
        
        streaming_city_weather_df = streaming_city_weather[(streaming_city_weather['date']>=lower_date_first_day.split(" ")[0]) & (streaming_city_weather['date']<=upper_date_seventh_day.split(" ")[0])]
        
        streaming_traffic_df = streaming_traffic[(streaming_traffic['date']>=lower_date_first_day.split(" ")[0]) & (streaming_traffic['date']<=upper_date_seventh_day.split(" ")[0])]
        
        print("streaming_routes_weather_df ",streaming_routes_weather_df.shape[0])
        print("streaming_city_weather_df ",streaming_city_weather_df.shape[0])
        print("streaming_traffic_df ",streaming_traffic_df.shape[0])
        
        cursor = connection.cursor()
        cursor.execute(
          "UPDATE constants SET weekly_streaming=(%s) WHERE id = 1", (int(weekly_streaming_df['weekly_streaming'][0])+1,));
        connection.commit()
        cursor.close()
        
    elif weekly_streaming_df['weekly_streaming'][0]==7:
        print("Week ended. New week will start from tomorrow.")
        cursor = connection.cursor()
        cursor.execute(
          "UPDATE constants SET weekly_streaming=(%s) WHERE id = 1", (1,));
        connection.commit()
        cursor.close()
    else:
        print(str(weekly_streaming_df['weekly_streaming'][0]), " day of the week")
        cursor = connection.cursor()
        cursor.execute(
          "UPDATE constants SET weekly_streaming=(%s) WHERE id = 1", (int(weekly_streaming_df['weekly_streaming'][0])+1,));
        connection.commit()
        cursor.close()
        
    # Setting flag for triggering weekly drift pipeline    
    if (not_15 == True) & (first_day_of_the_week == True):
        print("Weekly drift pipeline will trigger.")
        cursor = connection.cursor()
        cursor.execute(
          "UPDATE constants SET is_first_day_of_new_week=(%s) WHERE id = 1", (True,));
        connection.commit()
        cursor.close()
    else:
        cursor = connection.cursor()
        cursor.execute(
          "UPDATE constants SET is_first_day_of_new_week=(%s) WHERE id = 1", (False,));
        connection.commit()
        cursor.close()
        
    print("streaming_schedule_df - ", streaming_schedule_df.shape)
        
        
        
    print('search_date ', search_date)
    
    print('lower_date ', lower_date)
    
    print('upper_date ', upper_date)
    
    db_connection_str = 'mysql+pymysql://{username}:{password}@{endpoint}:{port_number}/{database_name}'
    db_connection = create_engine(db_connection_str)
    
    db_connection_str_pg = 'postgresql+psycopg2://{username}:{password}@{endpoint}:{port_number}/{database_name}'
    db_connection_pg = create_engine(db_connection_str_pg)
    
    # Append data to MySQL RDS table
    if first_day_of_the_week:
        streaming_city_weather_df.to_sql('city_weather', con=db_connection, if_exists='append', index=False)
        streaming_traffic_df.to_sql('traffic_details', con=db_connection, if_exists='append', index=False)
        streaming_routes_weather_df.to_sql('routes_weather', con=db_connection_pg, if_exists='append', index=False)
        print("streaming_routes_weather_df - ",streaming_routes_weather_df.shape)
        
        print("streaming_city_weather_df - ", streaming_city_weather_df.shape)
        
        print("streaming_traffic_df - ", streaming_traffic_df.shape)
        
        
    streaming_schedule_df = streaming_schedule_df.drop('delay', axis=1)
        
    streaming_schedule_df.to_sql('truck_schedule_data', con=db_connection, if_exists='append', index=False)
        
    cursor = connection.cursor()
    cursor.execute(
    "UPDATE constants SET day=(%s) WHERE id = 1", (((weekly_streaming_df['day'][0]) + timedelta(days=1)),));
    connection.commit()
    cursor.close()
    connection.close()
        
        
    # Update delay column of previous day
    if not_15 == True:
        print("IF")
        previous = (weekly_streaming_df['day'][0]) - timedelta(days=1)
        
        connection = pymysql.connect(
            user="",
            password="",
            host="",
            database="",
        )
        
        
        truck_schedule_data = pd.read_sql('select * from truck_schedule_data;', connection)
        
        truck_schedule_data['estimated_arrival'] = pd.to_datetime(truck_schedule_data['estimated_arrival'])
        
        lower_date = str(previous) + " 00:00:00"
        upper_date = str(previous) + " 23:59:59"
        
        print(lower_date)
        print(upper_date)
        
        truck_schedule_data = truck_schedule_data[(truck_schedule_data['estimated_arrival']>=lower_date) & (truck_schedule_data['estimated_arrival']<=upper_date)]
        
        truck_schedule_data.reset_index(drop=True, inplace=True)
        
        streaming_schedule = pd.read_csv(local_filename)
        
        streaming_schedule['estimated_arrival'] = pd.to_datetime(streaming_schedule['estimated_arrival'])
        
        streaming_schedule['departure_date'] = pd.to_datetime(streaming_schedule['departure_date'])
        
        row_counts = 0
        for i,d in truck_schedule_data.iterrows():
            delay = streaming_schedule[(streaming_schedule['truck_id']==d['truck_id']) & 
                               (streaming_schedule['route_id']==d['route_id']) & 
                               (streaming_schedule['departure_date']==d['departure_date'])].reset_index(drop=True)['delay'][0]
        
            cursor = connection.cursor()
            cursor.execute("UPDATE truck_schedule_data SET delay=(%s) WHERE truck_id = (%s) and route_id = (%s) and departure_date = (%s)",
                           (delay, d['truck_id'],d['route_id'],d['departure_date'],));
            connection.commit()
            cursor.close()
            row_counts=row_counts+1
        
        connection.close()
        print("Total number of rows updated are ", row_counts)

    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
