import csv
from kafka import KafkaProducer
import json
import time
import argparse


def publish_messages(csv_file, topic_name, bootstrap_servers, messages_per_second):
    # Create Kafka producer
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    # Read CSV file
    with open(csv_file, "r") as file:
        csv_reader = csv.DictReader(file)

        for row in csv_reader:
            # Convert row to JSON
            message = json.dumps(row)

            # Send message to Kafka topic
            producer.send(topic_name, value=row)

            # Print message for debugging
            print(f"Published: {message}")

            # Control publish rate
            time.sleep(1 / messages_per_second)

    # Ensure all messages are sent
    producer.flush()


if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Publish CSV data to Kafka topic with controllable rate."
    )
    parser.add_argument("csv_file", help="Path to the CSV file")
    parser.add_argument(
        "--topic_name", help="Name of the Kafka topic", default="fraud-detection"
    )
    parser.add_argument(
        "--bootstrap-servers",
        default="localhost:29092",
        help="Kafka bootstrap servers (default: localhost:29092)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=1.0,
        help="Publish rate in messages per second (default: 1.0)",
    )

    args = parser.parse_args()

    # Call the publish_messages function with parsed arguments
    publish_messages(
        args.csv_file, args.topic_name, [args.bootstrap_servers], args.rate
    )
