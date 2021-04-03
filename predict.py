import json
import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
import joblib
from kafka import KafkaConsumer, KafkaProducer, TopicPartition
from config import kafka_config
from PytorchDatasets import PandasDataset
from torch.utils.data import DataLoader


# Instantiate the Kafka Consumer object
consumer = KafkaConsumer(
    bootstrap_servers=kafka_config['servers'],
    group_id=None,
    enable_auto_commit=True)
    # value_deserializer=lambda x: json.loads(x.decode('utf-8')))

# Assign given TopicPartition to consumer
t_partition = TopicPartition(kafka_config['topics'][0], 0)
consumer.assign([t_partition])

# Seek to the most recent available offset
consumer.seek_to_end()

# Instantiate Kafka producer (to send predictions)
producer = KafkaProducer(bootstrap_servers=kafka_config['servers'],
    value_serializer=lambda x: json.dumps(x).encode('utf-8'))

# Specify list of input column names. Use all columns if None
feat_cols = None

# Provide the path to model's trained parameters (Pytorch)
# For sklearn models provide path/name of the model (must contain 'sklearn' keyword).
# For Spark ML model provide the list of paths to estimators (separate for each target variable)
# model_params = 'tabnet_params.pt'
# model_params = 'sklearn_random_forest'
model_params = ['spark_ml_models/model_latitude', 'spark_ml_models/model_longitude', 'spark_ml_models/model_geoAltitude']

# Provide the path for TabNet model's user-defined parameters (only for Pytorch TabNet model)
tabnet_user_params = 'tabnet_user_params.pickle'

# Load normalization parameters
with open(r"norm_params.pickle", "rb") as output_file:
    norm_params = pickle.load(output_file)

# Target's normalization parameters
y_min = np.array([norm_params[col]['min'] for col in norm_params['target']])
y_max = np.array([norm_params[col]['max'] for col in norm_params['target']])


def lla_to_ecef(df):
    """Convert WSG84 coordinates to cartesian ones
    
    """
    
    latitude = np.radians(df[0])
    longitude = np.radians(df[1])
    altitude = df[2]

    # WSG84 ellipsoid constants
    a = 6378137
    e = 8.1819190842622e-2

    # Prime vertical radius of curvature
    N = a / np.sqrt(1 - e**2 * np.sin(latitude)**2)
    
    x = (N + altitude) * np.cos(latitude) * np.cos(longitude)
    y = (N + altitude) * np.cos(latitude) * np.sin(longitude)
    z = ((1 - e**2) * N + altitude) * np.sin(latitude)

    df = np.hstack([np.expand_dims(x, axis=0), np.expand_dims(y, axis=0), np.expand_dims(z, axis=0)])
    
    return df


if isinstance(model_params, str):
    
    if model_params.endswith('.pt'):
        
        from TabNetBlocks import TabNet
        
        # Load TabNet user parameters
        with open(tabnet_user_params, 'rb') as file:
            tabnet_user_params = pickle.load(file)
        
        # Instantiate the Pytorch model
        model =  TabNet(tabnet_user_params['input_dim'],
                        tabnet_user_params['output_dim'],
                        tabnet_user_params['n_d'],
                        tabnet_user_params['n_a'],
                        gamma=tabnet_user_params['gamma'],
                        epsilon=tabnet_user_params['epsilon'],
                        n_shared=tabnet_user_params['n_shared'],
                        n_independent=tabnet_user_params['n_independent'],
                        n_steps=tabnet_user_params['n_steps'],
                        n_chunks=tabnet_user_params['n_chunks'],
                        chunk_size=tabnet_user_params['chunk_size'],
                        track_running_stats=tabnet_user_params['track_running_stats'],
                        momentum=tabnet_user_params['momentum'],
                        ghost_batch_norm=tabnet_user_params['ghost_batch_norm'])
        
        # Load model's trained parameters
        model.load_state_dict(torch.load(model_params))
            
        model.add_loss_fn(nn.MSELoss())
    
        optimizer = torch.optim.Adam(model.parameters(), lr=tabnet_user_params['learning_rate'])
        model.add_optimizer(optimizer)
        
        device = torch.device('cpu')
        model.add_device(device)
        
        # Set model to evaluation and double mode
        model.double() 
        model.eval()
        
        # Read evaluation_set 
        eval_df = pd.read_csv('round2_competition_data/eval_test/eval_test.csv', nrows=2e+6)
        
        
    elif 'sklearn' in model_params:
            
        est = joblib.load(model_params)
        
    else:
        raise TypeError("Unrecognized model")

elif isinstance(model_params, list):
    
    from pyspark.ml.regression import GBTRegressionModel
    from pyspark.sql import SparkSession
    from pyspark.sql import types
    from pyspark.sql import functions as F
    from pyspark.ml.feature import VectorAssembler
        
    spark = SparkSession.builder \
        .master("local[2]") \
        .appName("ads-b machine learning") \
        .config("spark.driver.memory", "6g") \
        .config("spark.jars", "jar_files/spark-sql-kafka-0-10_2.12-3.0.1.jar,"\
            "jar_files/kafka-clients-2.0.0.jar,"\
            "jar_files/spark-streaming-kafka-0-10-assembly_2.12-3.0.1.jar,"\
            "jar_files/spark-token-provider-kafka-0-10_2.12-3.0.1.jar,"\
            "jar_files/commons-pool2-2.6.2.jar") \
        .config("spark.driver.extraClassPath", "jar_files/spark-sql-kafka-0-10_2.12-3.0.1.jar,"\
            "jar_files/kafka-clients-2.0.0.jar,"\
            "jar_files/spark-streaming-kafka-0-10-assembly_2.12-3.0.1.jar,"\
            "jar_files/spark-token-provider-kafka-0-10_2.12-3.0.1.jar,"\
            "jar_files/commons-pool2-2.6.2.jar") \
        .getOrCreate()
        
    # Set number of output partitions
    spark.conf.set("spark.sql.shuffle.partitions", 5)
    
    # Set log level
    spark.sparkContext.setLogLevel("ERROR")
    
    target_cols = norm_params['target']
    models = {}
    
    for target in target_cols:

        model = [model for model in model_params if target in model]
        models[target] = GBTRegressionModel.load(model[0])
    
else:
    raise TypeError("Unrecognized model")


if not isinstance(model_params, list):
    
    if model_params.endswith('.pt') or 'sklearn' in model_params:
        
        for message in consumer:
            
            df = pd.read_json(message.value)

            timeAtServer = float(df.timeAtServer)
            aircraft = int(df.aircraft)

            if model_params.endswith('.pt'):
                if tabnet_user_params['feat_cols']:
                    x_cols = tabnet_user_params['feat_cols']
                else:
                    x_cols = norm_params['input_features']
            
            elif 'sklearn' in model_params:
                x_cols = norm_params['input_features']
                
            y_cols = norm_params['target']
            
            # Extract normalization parameters
            x_min = np.array([norm_params[col]['min'] for col in x_cols])
            x_max = np.array([norm_params[col]['max'] for col in x_cols])
        
            y_min = np.array([norm_params[col]['min'] for col in y_cols])
            y_max = np.array([norm_params[col]['max'] for col in y_cols])
            
            if model_params.endswith('.pt'):
                
                # Extent the batch with eval_df sample (batch_size should be > 1) to stablizie running std and var
                eval_df = eval_df.sample(n=1024*10-1)
                eval_set = PandasDataset(eval_df, norm_params, feat_cols=tabnet_user_params['feat_cols'])
                eval_loader = DataLoader(eval_set, batch_size=1024*10-1, shuffle=False, drop_last=True)
                
                # Normalize the real-time ADS-B data point
                input_feat = torch.DoubleTensor((np.array(df[x_cols]) - x_min) / (x_max - x_min))
                target = torch.DoubleTensor((np.array(df[y_cols]) - y_min) / (y_max - y_min))
        
                # Concatenate eval and real-time data points
                for eval_input, eval_target in eval_loader:
                    input_feat = torch.cat([input_feat, eval_input], dim=0)
                    target = torch.cat([target, eval_target], dim=0)
                    break
                
                input_feat.to(model.device)
                target.to(model.device)
                   
                # Forward through the network
                pred, _, _, _ = model.forward(input_feat)
                
                pred = pred[0]
                target = target[0]
                
                # Inverse the normalization
                pred = pred.detach().numpy() * (y_max - y_min) + y_min
                target = target.detach().numpy() * (y_max - y_min) + y_min
        
                pred_ecef = lla_to_ecef(pred)
                target_ecef = lla_to_ecef(target)
                
                # Calculate the average prediciton - target distance error in kilometers
                dist_error = np.abs(np.sqrt((pred_ecef[0] - target_ecef[0])**2 + (pred_ecef[1] - target_ecef[1])**2 + \
                        (pred_ecef[2] - target_ecef[2])**2) / 1000)
                        
                pred_json = {"pred": pred.tolist(), "target": target.tolist(), "dist_error": dist_error, 'timeAtServer': timeAtServer, 'aircraft': aircraft}
               
                producer.send(topic=kafka_config['topics'][1], value=pred_json)
                
        
            elif 'sklearn' in model_params:
                
                input_feat = (np.array(df[x_cols]) - x_min) / (x_max - x_min)
                target = (np.array(df[y_cols]) - y_min) / (y_max - y_min)
                
                pred = est.predict(input_feat)
         
                # Inverse the normalization
                pred = pred[0] * (y_max - y_min) + y_min
                target = target[0] * (y_max - y_min) + y_min
        
                pred_ecef = lla_to_ecef(pred)
                target_ecef = lla_to_ecef(target)
                
                # Calculate the average prediciton - target distance error in kilometers
                dist_error = np.sqrt((pred_ecef[0] - target_ecef[0])**2 + (pred_ecef[1] - target_ecef[1])**2 + \
                        (pred_ecef[2] - target_ecef[2])**2) / 1000
                
                dist_error = np.abs(dist_error)
                
                pred_json = {"pred": pred.tolist(), "target": target.tolist(), "dist_error": dist_error, 'timeAtServer': timeAtServer, 'aircraft': aircraft}
               
                producer.send(topic=kafka_config['topics'][1], value=pred_json)
                

if isinstance(model_params, list):
    
    schema_fields = types.StructType([types.StructField('timeAtServer', types.StructType([types.StructField("0", types.FloatType())])),
                                      types.StructField('aircraft', types.StructType([types.StructField('0', types.IntegerType())]))])
        
    for field in norm_params['input_features'] + norm_params['target']:
        if 'latitude' in field or 'longitude' in field or 'height_' in field:
            schema_fields.add(types.StructField(field, types.StructType([types.StructField('0', types.DoubleType())])))
        elif 'Altitude' in field or 'diff_' in field:
            schema_fields.add(types.StructField(field, types.StructType([types.StructField('0', types.FloatType())])))
        elif 'RSSI_' in field:
             schema_fields.add(types.StructField(field, types.StructType([types.StructField('0', types.ShortType())])))  
        elif 'tmp_' in field:
             schema_fields.add(types.StructField(field, types.StructType([types.StructField('0', types.LongType())]))) 
        elif 'mean' in field:
            schema_fields.add(types.StructField(field, types.StructType([types.StructField('0', types.DoubleType())])))
        else:
            schema_fields.add(types.StructField(field, types.StructType([types.StructField('0', types.FloatType())])))            
              
    # Define the json schema
    schema = types.StructType(schema_fields)
            
    df = spark \
       .readStream \
       .format("kafka") \
       .option("kafka.bootstrap.servers", ", ".join(kafka_config['servers'])) \
       .option("subscribe", kafka_config['topics'][0]) \
       .option("startingOffsets", "latest") \
       .option("failOnDataLoss", "false") \
       .load() \
       .selectExpr("CAST(value AS STRING)") \
       .select(F.from_json(F.col("value"), schema).alias("features")) \
       .select(*[F.col("features.{}.0".format(field)).alias(field) for field in norm_params['input_features'] + norm_params['target'] + ['timeAtServer', 'aircraft']]) \
       .withColumn('ID',  F.current_timestamp())
          
    # Write to console (for debug purposes)
    # df.printSchema() 
    # df.writeStream.outputMode("append").option("truncate", False).format("console").start().awaitTermination() 
      
    df_y = df.select('latitude', 'longitude', 'geoAltitude', 'ID')
    df_y = df_y \
        .withColumn("target", F.array('latitude', 'longitude', 'geoAltitude')) 
        #.drop('latitude', 'longitude', 'geoAltitude')
    
    df_x = df.drop('latitude', 'longitude', 'geoAltitude')
    
    vectorAssembler_x = VectorAssembler(inputCols=[col for col in norm_params['input_features']], outputCol='features')
    df_x = vectorAssembler_x.transform(df_x)
    df_x = df_x.select('features', 'ID', 'timeAtServer', 'aircraft')
    
    predictions = {}
    
    for target in target_cols:
                
        pred = models[target].transform(df_x)
        predictions[target] = pred.withColumnRenamed('prediction', 'pred_{}'.format(target))
        
    pred_df = predictions['latitude']
    
    # Join predictions into one data frame
    pred_df = pred_df \
        .join(predictions['longitude'], on='ID') \
        .drop('timeAtServer', 'aircraft') \
        .join(predictions['geoAltitude'], on='ID') \
        .drop('features')
        
    pred_df = pred_df \
        .withColumn("pred", F.array('pred_latitude', 'pred_longitude', 'pred_geoAltitude')) 
        #.drop('pred_latitude', 'pred_longitude', 'pred_geoAltitude')
        
    pred_df = pred_df.join(df_y, on='ID')

    # Calculation of the distance error
    pred_df = pred_df \
        .withColumn('N1', 6378137 / (F.sqrt(1 - 8.1819190842622e-2**2 * F.sin(F.radians(F.col('latitude'))**2)))) \
        .withColumn('N2', 6378137 / (F.sqrt(1 - 8.1819190842622e-2**2 * F.sin(F.radians(F.col('pred_latitude'))**2))))

    pred_df = pred_df \
        .withColumn('x1', (F.col('N1') + F.col('geoAltitude')) * F.cos(F.radians(F.col('latitude'))) * F.cos(F.radians(F.col('longitude')))) \
        .withColumn('y1', (F.col('N1') + F.col('geoAltitude')) * F.cos(F.radians(F.col('latitude'))) * F.sin(F.radians(F.col('longitude')))) \
        .withColumn('z1', ((1 - 8.1819190842622e-2**2) * F.col('N1') + F.col('geoAltitude')) * F.sin(F.radians(F.col('latitude')))) \
        .withColumn('x2', (F.col('N2') + F.col('pred_geoAltitude')) * F.cos(F.radians(F.col('pred_latitude'))) * F.cos(F.radians(F.col('pred_longitude')))) \
        .withColumn('y2', (F.col('N2') + F.col('pred_geoAltitude')) * F.cos(F.radians(F.col('pred_latitude'))) * F.sin(F.radians(F.col('pred_longitude')))) \
        .withColumn('z2', ((1 - 8.1819190842622e-2**2) * F.col('N2') + F.col('pred_geoAltitude')) * F.sin(F.radians(F.col('pred_latitude'))))

    pred_df = pred_df \
        .withColumn('dist_error', F.sqrt((F.col('x1') - F.col('x2'))**2 + \
                                           (F.col('y1') - F.col('y2'))**2 + \
                                           (F.col('z1') - F.col('z2'))**2) / 1000) \
        .drop('latitude', 'longitude', 'geoAltitude') \
        .drop('pred_latitude', 'pred_longitude', 'pred_geoAltitude') \
        .drop('N1', 'N2', 'x1', 'y1', 'z1',  'x2', 'y2', 'z2')

    # Write stream to console for debug purposes
    # pred_df.writeStream.outputMode("append").option("truncate", False).format("console").start().awaitTermination() 

    # Write stream to Kafka
    pred_df = pred_df \
        .select(F.to_json(F.struct("pred", "target", "dist_error", "timeAtServer", "aircraft")).alias("value")) 

    pred_df \
       .writeStream \
       .format("kafka") \
       .option("kafka.bootstrap.servers", ", ".join(kafka_config['servers'])) \
       .option("topic", kafka_config['topics'][1]) \
       .option("checkpointLocation", "checkpoint") \
       .start() \
       .awaitTermination()
