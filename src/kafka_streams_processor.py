from confluent_kafka.streams import StreamsBuilder, KafkaStreams
from confluent_kafka import KafkaError
import json

def create_topology():
    builder = StreamsBuilder()
    
    # Create a stream from the input topic
    input_stream = builder.stream("mysql-data")
    
    # Process the stream
    processed_stream = input_stream.map(lambda k, v: (k, json.loads(v).upper()))
    
    # Write the processed stream to the output topic
    processed_stream.to("processed-data")
    
    return builder.build()

def run_streams():
    topology = create_topology()
    
    streams = KafkaStreams(topology, {
        'bootstrap.servers': 'kafka:9092',
        'application.id': 'kafka-streams-processor'
    })
    
    streams.start()
    
    try:
        while True:
            pass
    except KeyboardInterrupt:
        pass
    finally:
        streams.close()

if __name__ == "__main__":
    run_streams()
