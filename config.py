"""Configuration file.

"""

mariadb_hostname = 'localhost'
mariadb_port = 3306
mariadb_database_name = 'adsb'
mariadb_table_name = 'adsb'

kafka_config = {'servers': ['localhost:9092'], 'topics': ['adsb', 'adsb-pred']}