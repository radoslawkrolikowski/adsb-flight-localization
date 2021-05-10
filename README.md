# ADS-B Flight Localization

This project will guide you from the beginning with the data inspection and preprocessing up to crafting an end to end application for aircraft localization based on crowdsourced air traffic control communication data. The dataset is a part of the Aircraft Localization Competition powered by **OpenSky Network** and **Cyber-Defence Campus - armasuisse Science and Technology**. It contains the ADS-B transmissions collected by the large-scale sensor network and poses the following challenges:
- volume - perform data preprocessing and training of the ML models on the data that doesn't fit into the memory,
- velocity - real-time data preprocessing, prediction and visualization,
- veracity - issue of unsynchronized receivers, incorrect sensors' localizations,
- value - perform data preprocessing and predictive analytics that leads to insights - prediction of the aircraft current coordinates and altitude,
- variety - extraction of the data from the JSON arrays nested inside the table.

To ensure that our application meets the scalability and performance requirements we will have to use the appropriate technologies. The following are the tools that are going to be utilized:
- distributed data preprocessing with *Apache Spark* and *Modin*,
- use of ensemble methods (*Apache Spark ML, Sklearn*) and *TabNet* model (*Pytorch*) for tabular learning,
- memory-efficient loading of data thanks to custom *Pytorch Datasets* implementations,
- utilize *Apache Kafka* to stream real-time data between internal components of the application,
- real-time data visualization with *Flask* and *Leaflet.js*.

Additional information about the Aircraft Localization Competition can be found on the official website - [AIcrowd](https://www.aicrowd.com/challenges/cyd-campus-aircraft-localization-competition#introduction).


### Architecture

![Architecture](https://github.com/radoslawkrolikowski/adsb-flight-localization/blob/main/assets/architecture.png)


### Demo

![demo gif](https://github.com/radoslawkrolikowski/adsb-flight-localization/blob/main/assets/demo.gif)


### Table of contents

* [data_inspection](https://nbviewer.jupyter.org/github/radoslawkrolikowski/adsb-flight-localization/blob/main/data_inspection.ipynb)
 	
    The data inspection and visualization notebook will guide you through the process of loading the data, examining the distribution of the features and visualizing an example flight in conjunction with recorded flight parameters such as timestamp, timeAtServer, received signal strength indicator (RSSI), barometric and GPS altitude.

* [data_preprocessing](https://nbviewer.jupyter.org/github/radoslawkrolikowski/adsb-flight-localization/blob/main/data_preprocessing.ipynb)

   The second tutorial contains instructions on how to perform the data preprocessing that consist of the following steps:

   - drop the duplicated rows,
   - check the data frame in terms of missing values,
   - explode the measurements JSON array, sort it according to sensor serial number and limit the number of measurements,
   - extract the sensor, timestamp and RSSI information from an array of measurements,
   - verify sensors' location and correct their elevation,
   - perform data casting and filtering,
   - conduct timestamps synchronization,
   - create linear regression models of timestamp corrections,
   - perform the feature extraction,
   - calculate the normalization parameters,
   - save preprocessed data to *HDF5* or *MariaDB*.

* [prepare_eval_test_datasets](https://nbviewer.jupyter.org/github/radoslawkrolikowski/adsb-flight-localization/blob/main/prepare_eval_test_datasets.ipynb)

   In this notebook, we will conduct the data preprocessing to make the evaluation and test datasets ready.  

* [training_ensemble](https://nbviewer.jupyter.org/github/radoslawkrolikowski/adsb-flight-localization/blob/main/training_ensemble.ipynb)

   In this notebook, we are going to build the following estimators from the training set:

   - Random forest regressor (*Sklearn*)
   - Gradient-boosted trees (*Apache Spark*)

   After training the ensemble models will be assessed on an evaluation set.

* [training_TabNet](https://nbviewer.jupyter.org/github/radoslawkrolikowski/adsb-flight-localization/blob/main/training_TabNet.ipynb)

   In this notebook, we are going to train the TabNet neural network model. The implementation of all building blocks of the model can be found in the file [TabNetBlocks](https://github.com/radoslawkrolikowski/adsb-flight-localization/blob/main/TabNetBlocks.py) in this repository.

* [TabNetBlocks](https://github.com/radoslawkrolikowski/adsb-flight-localization/blob/main/TabNetBlocks.py)

   This file contains the *Pytorch* implementations of the following architectures and tools:
   - TabNet neural network model according to: <https://arxiv.org/pdf/1908.07442.pdf>
   - Attentive Transformer
   - Feature Transformer
   - Ghost Batch Normalization
   - Sparsemax activation function
   - Gated Linear Unit blocks
   - Shared and dependant GLU fully connected layers across all decision steps

* [PytorchDatasets](https://github.com/radoslawkrolikowski/adsb-flight-localization/blob/main/PytorchDatasets.py)

   The Implementation of the custom Pytorch Datasets that can be used to load the data from *HDF5*, *Pandas* or *MariaDB*, but also to perform data normalization.

* [createDB](https://github.com/radoslawkrolikowski/adsb-flight-localization/blob/main/createDB.py)

   Creates an 'adsb' database that stores in the main table the preprocessed training dataset.

* [config](https://github.com/radoslawkrolikowski/adsb-flight-localization/blob/main/config.py)
 	
   The configuration file that includes: 
   - Kafka brokers addresses and topics
   - Database (*MariaDB*) properties

* [ADSB_producer](https://github.com/radoslawkrolikowski/adsb-flight-localization/blob/main/ADSB_producer.py)

   The producer simulates the stream of ADS-B data. It allows specifying the frequency of the messages and the data filtering parameters such as aircraft serial, its localization or altitude. The raw, real-time ADS-B data is preprocessed according to the same steps that have been taken during the training set preparation. Subsequently, that data is published to corresponding Kafka topic, so that we can use it to make a real-time prediction and visualization of the aircraft position.

* [ADSB_preprocessing](https://github.com/radoslawkrolikowski/adsb-flight-localization/blob/main/ADSB_preprocessing.py)

   Performs the ADS-B data preprocessing that includes:
   - exploding the measurements JSON array,
   - extracting the sensor, timestamp and RSSI information from an array of measurements,
   - conducting timestamps synchronization,
   - adding the sensor localization data,
   - performing the feature extraction,
   - filling missing values.

* [predict](https://github.com/radoslawkrolikowski/adsb-flight-localization/blob/main/predict.py)

   - Subscribes to a real-time stream of records in given Kafka topic
   - Performs real-time data normalization and prediction using one of the available models:
      - *Pytorch* TabNet
      - *Apache Spark* Gradient-boosted trees 
      - *Sklearn* Random forest regressor
   - Calculates the average prediction-target distance error in kilometres
   - Sends the predictions, targets, distance error, timeAtServer and aircraft serial number to the Kafka topic

* [flights_map](https://github.com/radoslawkrolikowski/adsb-flight-localization/blob/main/flights_map.py)

   The real-time flight radar map developed using the *Flask* web framework, *leaflet.js*, *chart.js* and *JavaScript*. The index.html file can be found in the `templates` directory - [here](https://github.com/radoslawkrolikowski/adsb-flight-localization/blob/main/templates/index.html). The `static` directory should contain the following files: CSS, chart.js, leaflet-hotline and leaflet-rotatedmarker files as well as the logo and the plane icon.

   The ADS-B Flight Radar can be accessed under the following URL in your browser - http://localhost:5001/

   You can click on the plane icon to visualize its route and depict the altitude graph.


### Dataset

Dataset can be downloaded from the Aircraft Localization Competition official website - [AIcrowd](https://www.aicrowd.com/challenges/cyd-campus-aircraft-localization-competition#data-sets).

Dataset folder structure is as follows:

	round2_training/
		├── round2/
			├── round2_training1.csv
			├── round2_training2.csv 
			├── round2_training3.csv 
			├── round2_sensors.csv 

	round2_competition_data/
		├── round2_competition.csv
		|── round2_sensors.csv 

More detailed background information on the provided data can be found [here](https://competition.opensky-network.org/documentation.html)


### Docker

1. Install Docker for your system - <https://docs.docker.com/get-docker/>
2. Create a directory for mysql data persisted by Docker:
   - `cd adsb-flight-localization`
   - `mkdir mysql`
3. Change the *MariaDB* and *Kafka* configuration in *config.py*:
   - `mariadb_hostname = 'mariadb'`
   - `kafka_config = {'servers': ['kafka:9092']}`
4. Build and run the Docker containers:

   Change directory to `docker`:
 
      - `cd docker`

   Set `START_RADAR='true'` if you want to run the ADSB producer, perform the aircraft localization prediction and launch the flights_map *Flask* application while starting the Docker containers, otherwise set `START_RADAR='false'`

   Start the Docker containers without running the ADS-B Flight-Radar, for example, to perform data preprocessing or model training:
   
      - `START_RADAR='false' docker compose up`
   
   Start the Docker containers and the ADS-B Flight-Radar:
      
      - `START_RADAR='true' docker compose up`

   If you are starting it for the first time, the `docker compose up` command begins with building the containers from specified images and Dockerfiles. This process might be compute-intensive, thus if you are experiencing issues (OOM), try to build the jupyter-spark container on its own by executing the following command:
      
      - `cd adsb-flight-localization`
      - `docker build . -t jupyter-spark:1.0 -f docker/jupyter-spark/Dockerfile`

   You can access the Jupyter Notebook (running in Docker) by opening the following URL in your browser (host): <http://localhost:8888>. If you are asked about the access token, copy it from the console. ADS-B Flight-Radar can be accessed by opening <http://localhost:5001>.


### Installing

#### JAVA 8
Apache Spark and Kafka run on JAVA 8/11. Hence, we will start by installing the Java SE Development Kit 8:
1. Download the JDK from the official site - <https://www.oracle.com/uk/java/technologies/javase/javase-jdk8-downloads.html>:
2. Create the directory for JDK:
 - `sudo mkdir /usr/lib/jvm`
3. Extract the JDK repository:
- `cd /usr/lib/jvm`
- `sudo tar -xvzf jdk-8u281-linux-x64.tar.gz`
4. Set $JAVA_HOME environmental variable in .bashrc file:
   - `export JAVA_HOME='/usr/lib/jvm/jdk1.8.0_281'`
5. Verify the version of the JDK with the following command:
- `java -version`

#### Apache Spark:
1. Download Apache Spark from <https://spark.apache.org/downloads.html>
2. Go to the directory where spark zip file was downloaded and unpack it:
   - `tar -zxvf spark-3.0.0-bin-hadoop2.7.tgz`
3. In .bashrc file configure other environmental variables for Spark:
   - `export SPARK_HOME='spark-3.0.0-bin-hadoop2.7'`
   - `export PATH=$SPARK_HOME:$PATH`
   - `export PATH=$PATH:$SPARK_HOME/bin`
   - `export PYTHONPATH=$SPARK_HOME/python;%SPARK_HOME%\python\lib\py4j-0.10.7-src.zip:%PYTHONPATH%`
   - `export PYSPARK_DRIVER_PYTHON="python" `
   - `export PYSPARK_PYTHON=python3`
   - `export SPARK_YARN_USER_ENV=PYTHONHASHSE`

#### Apache Kafka
1. Donwload Kafka:
   - `wget https://downloads.apache.org/kafka/2.7.0/kafka_2.12-2.7.0.tgz`
2. Unpack Kafka repository:
   - `tar -xvf kafka_2.12-2.7.0.tgz`
3. Create a symbolic link:
   - `ln -s kafka_2.12-2.7.0 kafka`

Setting up a multi-broker cluster:
1. Create a config file for each of the brokers using sample properties:
   - `cd kafka_2.12-2.7.0`
   - `cp config/server.properties config/server-1.properties`
   - `cp config/server.properties config/server-2.properties`
2. Now edit these new files and set the following properties:
	
    config/server-1.properties:
	`delete.topic.enable=true
        broker.id=1
        listeners=PLAINTEXT://:9093
        log.dirs=/tmp/kafka-logs-1`
 
    config/server-2.properties:
	`delete.topic.enable=true
        broker.id=2
        listeners=PLAINTEXT://:9094
        log.dirs=/tmp/kafka-logs-2`

#### MariaDB
1. Update the `apt` packages index:
   - `sudo apt update`
2. Install MariaDB by running the following command:
   - `sudo apt install mariadb-server`

#### Python packages:
Install all packages included in requirements.txt

1. Create a virtual environment (conda, virtualenv etc.).
   - `conda create -n <env_name> python=3.7`
2. Activate your environment.
   - `conda activate <env_name>`
3. Install requirements.
   - `pip install -r requirements.txt `
4. Restart your environment.
    - `conda deactivate`
    - `conda activate <env_name>`

#### Node.js
1. To install Node.js run the following commands:
   - `curl -fsSL https://deb.nodesource.com/setup_14.x | bash -`
   - `apt-get install -y nodejs`
2. Verify that the Node.js and `npm` were successfully installed:
   - `node --version`
   - `npm --version`

#### Leaflet.js
1. Install Leaflet.js using `npm` package manager:
   - `npm install leaflet`
2. You will find the Leaflet release files in node_modules/leaflet/dist.
3. To extend the Leaflet.js capabilities we will install two additional plugins:
   - `npm install leaflet-hotline`
   - `npm install leaflet-rotatedmarker`

#### Chart.js
1. Use `npm` to install Chart.js:
   - `npm install chart.js`

#### Dependencies
All indispensable JAR files can be found in jar_files directory.


### Usage

A. Data inspection and preprocessing as well as training of the ML models.
1. Specify your configuration by modifying config.py file:
   - MariaDB properties
   - Kafka brokers addresses and topics
2. Run and follow the [data_inspection](https://nbviewer.jupyter.org/github/radoslawkrolikowski/adsb-flight-localization/blob/main/data_inspection.ipynb) notebook to get an insight into the nature of the data.
3. Create the MariaDB database by running the createDB.py script (not necessary if you want to store preprocessed data in the HDF5 file)
4. Use the [data_preprocessing](https://nbviewer.jupyter.org/github/radoslawkrolikowski/adsb-flight-localization/blob/main/data_preprocessing.ipynb) notebook to perform the preprocessing of the entire training dataset (consists of 3 files).
5. Run the [prepare_eval_test_datasets](https://nbviewer.jupyter.org/github/radoslawkrolikowski/adsb-flight-localization/blob/main/prepare_eval_test_datasets.ipynb) notebook to make the evaluation and test sets ready.
6. Run the [training_ensemble](https://nbviewer.jupyter.org/github/radoslawkrolikowski/adsb-flight-localization/blob/main/training_ensemble.ipynb) notebook to build the Random forest regressor and the Gradient-boosted trees estimators from the training set:
7. Use the [training_TabNet](https://nbviewer.jupyter.org/github/radoslawkrolikowski/adsb-flight-localization/blob/main/training_TabNet.ipynb) notebook to train the TabNet neural network model.

B. Real-time data preprocessing, prediction and visualization.
1. Before each run of the application we have to start the ZooKeeper and Kafka brokers:

    1. Start the ZooKeeper:
        - `cd zookeeper/`
        - `bin/zkServer.sh start conf/zookeeper.properties`
    2. Check if it started correctly:
        - `bin/zkServer.sh status conf/zookeeper.properties`

    3. Start the Kafka nodes:
       - `cd kafka/`
       - `bin/kafka-server-start.sh config/server.properties`
       - `bin/kafka-server-start.sh config/server-1.properties`
       - `bin/kafka-server-start.sh config/server-2.properties`

2. Create the Kafka topics if you run the application for the first time (list of sample topics can be found in config.py file):
 
 	1. Create topic:
		- `bin/kafka-topics.sh --create --bootstrap-server localhost:9092 --replication-factor 3 --partitions 1 --topic topic_name`
 
	2. List available topics:
		- `bin/kafka-topics.sh --list --bootstrap-server localhost:9092`

3. Run the [flights_map](https://github.com/radoslawkrolikowski/adsb-flight-localization/blob/main/flights_map.py) Flask application and then go to the http://localhost:5001/ to access the map.  
4. Then we can run the ADSB_producer.py to preprocess and publish the real-time ADS-B data to the Kafka topic.
5. To make a real-time prediction run predict.py file (only data that comes after predict.py is launched is going to be considered).
6. Observe the real-time aircraft localization predictions using the Flight Radar map (http://localhost:5001/). You can click on the plane icon to visualize its route and depict the altitude graph.
 

### References

* <https://www.aicrowd.com/challenges/cyd-campus-aircraft-localization-competition>
* <https://competition.opensky-network.org/documentation.html>
* <https://pytorch.org/docs/stable/index.html>
* <https://spark.apache.org/docs/3.0.0/api/python/index.html>
* <https://kafka.apache.org/21/documentation.html>
* <https://flask.palletsprojects.com/en/1.1.x/>
* <https://leafletjs.com/reference-1.7.1.html>
* <https://arxiv.org/pdf/1908.07442v1.pdf>
* <https://www.researchgate.net/publication/304584658_The_Testing_of_MLAT_Method_Application_by_means_of_Usage_low-cost_ADS-B_Receivers>
* <https://www.lenders.ch/publications/reports/arxiv16_2.pdf>
