"""Configuration file.

"""

# If using Docker use the mariadb_hostname = 'mariadb', otherwise use 'localhost'
mariadb_hostname = 'localhost' # 'mariadb'
mariadb_port = 3306
mariadb_database_name = 'adsb'
mariadb_table_name = 'adsb'
mariadb_user = '<user_name>'
mariadb_password = '<your_password>'

kafka_config = {'servers': ['localhost:9092'], 'topics': ['adsb', 'adsb-pred']}

# For Docker use the following Kafka configuration (uncomment)
# kafka_config = {'servers': ['kafka:9092'], 'topics': ['adsb', 'adsb-pred']}