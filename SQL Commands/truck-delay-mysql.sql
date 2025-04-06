# SQL commands

# Selecting all columns and top 10 rows from table city_weather from truckDB database
select * from truckDB.city_weather limit 10;

# Selecting all columns and top 10 rows from table drivers_details from truckDB database
select * from truckDB.drivers_details limit 10;

# Selecting all columns and top 10 rows from table drivers_details from truckDB database
select * from truckDB.drivers_details limit 10;

# Selecting all columns and top 10 rows from table truck_details from truckDB database
select * from truckDB.truck_details limit 10;

# Selecting all columns and top 10 rows from table truck_schedule_data from truckDB database
select * from truckDB.truck_schedule_data limit 10;

#--------------------------------------------------------------------------------------------------
# Driver details analysis


# Driver Age Statistics Summary
select 
	count(*) As total_drivers,
	avg(age) As avg_age,
    min(age) As min_age,
    max(age) As max_age
from truckDB.drivers_details;

# Driver Ratings Summary
select 
	avg(ratings) As avg_rating,
    min(ratings) As min_rating,
    max(ratings) As max_rating
from truckDB.drivers_details;

# Driver Gender Distribution
select gender, COUNT(*) as count_gender from truckDB.drivers_details Group By gender;

# Driver Ratings Distribution
select ratings, COUNT(*) as count_ratings from truckDB.drivers_details Group By ratings;

#-------------------------------------------------------------------------------------------
#truck detail table analysis

# Selecting all columns and top 10 rows from table truck_details from truckDB database
select * from truckDB.truck_details limit 10;

# Total Trucks
select count(*) As total_truck from truckDB.truck_details;

# Truck Age Statistics Summary
select 
	min(truck_age) As min_truck_age,
	max(truck_age) As max_truck_age,
    avg(truck_age) As avg_truck_age
from truckDB.truck_details;

# Truck Fuel Type Distribution
select fuel_type, COUNT(*) as count_fuel_type from truckDB.truck_details Group By fuel_type;

# To view all column names from respective tables
show columns from truckDB.truck_details;
show columns from truckDB.drivers_details;

# Driver Ratings and Average Speed Comparison
select ratings, average_speed_mph from truckDB.drivers_details order by ratings;

#-------------------------------------------------------------------------------------------
# Traffic details analysis

# Selecting all columns and top 10 rows from table traffic_details from truckDB database
select * from truckDB.traffic_details limit 10;

# List of Distinct Route IDs in Traffic Details
select distinct route_id from truckDB.traffic_details;

# Total Distinct Routes in Traffic Details
select count(distinct route_id) as total_routes from truckDB.traffic_details;

# View columns of traffic details table
show columns from truckDB.traffic_details;

# Total Traffic Details Recorded
select count(*) as total_traffic_details from truckDB.traffic_details;

# Traffic Accident Summary
select accident, count(accident) as is_accident from truckDB.traffic_details Group By accident;

# Traffic Data Date Range
select min(date) as min_date, max(date) as max_date from truckDB.traffic_details;

#-------------------------------------------------------------------------------------------
# City Weather Analysis

# Selecting all columns and top 15 rows from table traffic_details from truckDB database
select * from truckDB.city_weather limit 15;

# Total Rows in City Weather Data
select count(*) as total_rows from truckDB.city_weather;

# Total Distinct Cities in City Weather Data
select count(distinct city_id) as total_cities from truckDB.city_weather;

# City Weather Data Date Range
select min(date) as min_date, max(date) as max_date from truckDB.city_weather;

#-------------------------------------------------------------------------------------------
# Truck schedule data analysis

# Selecting all columns and top 10 rows from table truck_schedule_data from truckDB database
select * from truckDB.truck_schedule_data limit 10;

# Total Rows in Truck Schedule Data
select count(*) total_rows from truckDB.truck_schedule_data;

# Total Distinct Trucks in Truck Schedule Data
select count(distinct truck_id) as total_trucks from truckDB.truck_schedule_data;

# Delayed Trucks Summary
select delay, COUNT(*) as total_delayed_trucks from truckDB.truck_schedule_data Group By delay;

# Distinct Departure Dates in Truck Schedule Data
select distinct departure_date as total_dates from truckDB.truck_schedule_data order by departure_date;






