import mariadb
import sys
import pickle
import logging
from getpass import getpass
from config import mariadb_hostname, mariadb_database_name, mariadb_port, mariadb_table_name, mariadb_user, mariadb_password


# Set logging level
logging.basicConfig(level=logging.INFO)

# Connect to MariaDB server
try:
    cnx = mariadb.connect(host=mariadb_hostname, port=mariadb_port, user=mariadb_user, password=mariadb_password)
except mariadb.Error as msg:
    logging.error("Error connecting to MariaDB Platform: {}".format(msg))
    sys.exit(1)

# Instantiate the cursor object
cursor = cnx.cursor()

# Create database <mariadb_database_name> if not exists
try:
    cursor.execute("CREATE DATABASE IF NOT EXISTS {};".format(mariadb_database_name))
    logging.info('Database {} has been created'.format(mariadb_database_name))
    
    # Use given database
    cursor.execute("USE {};".format(mariadb_database_name))

except mariadb.Error as msg:
    logging.error(msg)

# Get the columns to create (from norm_params.pickle)
with open(r"norm_params.pickle", "rb") as output_file:
    norm_params = pickle.load(output_file)
    
columns = ['timeAtServer', 'aircraft'] + norm_params['input_features'] + norm_params['target']

# Specify the table schema
fields = ''

for field in columns:
    if field in ['longitude', 'latitude', 'mean_lat', 'mean_lon', 'w_mean_lat', 'w_mean_lon']:
        fields += ', {} FLOAT(8,5)'.format(field)
    elif 'Altitude' in field:
        fields += ', {} FLOAT(7,2)'.format(field)
    elif 'tmp' in field:
        fields += ', {} BIGINT'.format(field)
    elif 'RSSI' in field:
        fields += ', {} SMALLINT'.format(field)
    elif 'height' in field:
        fields += ', {} FLOAT(7,2)'.format(field)
    elif 'diff' in field:
        fields += ', {} BIGINT'.format(field)
    elif 'timeAtServer' in field:
        fields += ', {} FLOAT(7,3)'.format(field)
    else:
        fields += ', {} INT'.format(field)
    
# Create the table
create_table = 'CREATE TABLE IF NOT EXISTS ' + mariadb_table_name  + ' (ID INT KEY AUTO_INCREMENT' + fields + ");"

try:
    cursor.execute(create_table)
    logging.info('Table {} has been created'.format(mariadb_table_name))
except mariadb.Error as msg:
    logging.error(msg)
    
# Commit changes
cnx.commit()
