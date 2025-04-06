# SQL commands

# Selecting all columns and top 10 rows from table routes_details
select * from routes_details limit 10;

# Total Rows in Routes Details
select count(*) as total_rows from routes_details;

# Route Details Summary
select count(distinct route_id) as total_routes,
count(distinct origin_id) as total_origins,
count(distinct destination_id) as total_destination
from routes_details;

# Route Distance Range
select
	min(distance) as min_distance,
	max(distance) as max_distance
from routes_details;	

# Route Average Travel Time Range
select
	min(average_hours) as min_hrs,
	max(average_hours) as max_hrs
from routes_details;	

#--------------------------------------------------------------------------
# routes weather analysis

# Selecting all columns and top 10 rows from table routes_details
select * from routes_weather limit 10;

# Total Rows in Routes Weather Data
select count(*) as total_rows from routes_weather;

# Total Distinct Routes in Routes Weather Data
select count(distinct route_id) as total_distinct_routes from routes_weather;

# Columns in Routes Weather Table
select column_name from information_schema.columns where table_name = 'routes_weather';

# Count of Distinct Chance of Thunder Entries in Routes Weather Data
select count(distinct chanceofthunder) from routes_weather;

# Correlation Between Temperature and Pressure in Routes Weather Data
select corr(temp,pressure) as cor_temp_pressure from routes_weather;

