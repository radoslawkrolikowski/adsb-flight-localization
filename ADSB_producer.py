import pandas as pd
import time
from kafka import KafkaProducer
from config import kafka_config
from ADSB_preprocessing import preprocess_adsb

# Specify the chunk size and the number of rows to be read from the CSV dataset
chunk_size = 1000
nrows = 10000

# Message frequency in seconds, if None use the timeAtServer
mssg_freq = 10

# Specify the timeAtServer scale factor
# Used to increase/decrease the frequency of messages (scaled timeAtServer frequency)
timeAtServer_scale = 1

# Specift the maximum number of measurements
max_measurements = 6

# Specify the number of measurements to be taken into account while calculating differences
n_meas_diff = 4

# Whether to only use data without target variables (lat, lon, geoAltitude)
use_data_without_targets = False 

# Specify the data filtering parameters, otherwise set to None
# Filter the dataset in terms of aircraft number, its localization or altitude
aircraft = 863 # [181, 843] single int or list of aircraft serials
lat_range = None # [42.0, 45.0] # latitude degree range
lon_range = None # [0, 0.8] # longitude degree range
baro_alt_range = None #[0, 10500] # baroAltitude range in meters

# Read the competition (evaluation/test) dataset that will simulate the ads-b transmission
data = pd.read_csv('round2_competition_data/round2_competition.csv', chunksize=chunk_size, nrows=nrows)

# Instantiate ads-b data Kafka producer
producer = KafkaProducer(bootstrap_servers=kafka_config['servers'])
    
prevTimeAtServer = 0

for chunk in data:
        
    # Perform data filtering
    if aircraft:
        if isinstance(aircraft, int):
            chunk = chunk.loc[chunk.aircraft == aircraft]
        if isinstance(aircraft, list):
            chunk = chunk.loc[(chunk.aircraft.isin(aircraft))]
            
    if baro_alt_range:
        chunk = chunk.loc[chunk.baroAltitude.between(*baro_alt_range)]
        
    if use_data_without_targets:
        chunk = chunk.loc[chunk.latitude.isna()]
    else:
        if lat_range:
            chunk = chunk.loc[chunk.latitude.between(*lat_range)]
        if lon_range:
            chunk = chunk.loc[chunk.longitude.between(*lon_range)]
    
    for i, row in chunk.iterrows():

        process_start_time = time.time()
 
        timeatserver = row['timeAtServer']    
        aircraft_serial = row['aircraft']

        # Perform the ads-b data preprocessing
        row_preprocessed = preprocess_adsb(row, max_measurements, n_meas_diff)

        # Add timeAtServer and aircraft serial information
        row_preprocessed['timeAtServer'] = timeatserver
        row_preprocessed['aircraft'] = aircraft_serial

        # Convert pandas DataFrame to json
        row_json = row_preprocessed.to_json().encode('utf-8')
        
        # Send data to Kafka topic
        producer.send(topic=kafka_config['topics'][0], value=row_json)
        
        process_end_time = time.time()
        process_time = process_end_time - process_start_time
        
        if not mssg_freq:
            mssg_freq =  timeAtServer * timeAtServer_scale - prevTimeAtServer
            prevTimeAtServer = timeAtServer * timeAtServer_scale
        
        time.sleep(mssg_freq - process_time)
     