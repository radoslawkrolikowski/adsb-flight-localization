#!/bin/bash

echo "Starting $0 script"

if [ ! -d /var/lib/mysql/adsb ];
then
    echo Creating the MariaDB database
    python3 createDB.py
else
    echo Database exists
fi


if ( "${START_RADAR}" == "true" );
then
    # Wait for Kafka to start up
    sleep 30

    echo Starting flights_map.py &
    python3 flights_map.py &
    sleep 3 &
    echo Starting predict.py &
    python3 predict.py &
    sleep 15 &
    echo Starting ADSB_producer.py &
    python ADSB_producer.py

else
    echo Flight Radar not started
fi