from flask import Flask, render_template, Response, stream_with_context
from kafka import KafkaConsumer, TopicPartition
from config import kafka_config
import json


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/adsb/')
def get_messages():

    # Instantiate Kafka Consumer object
    consumer = KafkaConsumer(
        bootstrap_servers=kafka_config['servers'],
        value_deserializer=lambda x: json.loads(x.decode('utf-8')))

    # Assign given TopicPartition to consumer
    t_partition = TopicPartition(kafka_config['topics'][1], 0)
    consumer.assign([t_partition])

    # Seek to the most recent available offset
    consumer.seek_to_end()

    def events():

        for message in consumer:

            # Transform message to event-stream format
            message = 'data:{}\n\n'.format(message.value)

            yield message.replace('\'', '"')
            
    return Response(events(), mimetype='text/event-stream')


if __name__ == '__main__':
    
    app.run(debug=True, host='0.0.0.0', port=5001)
