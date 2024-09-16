import csv
from kafka import KafkaProducer
from kafka.errors import KafkaError
import json
import time
import argparse
import sys

sys.path.append("../")

from pydantic import ValidationError
from models.transaction import TransactionModel


def check_kafka_connection(bootstrap_servers):
    try:
        producer = KafkaProducer(bootstrap_servers=bootstrap_servers)
        producer.close()
        print("Successfully connected to Kafka")
        return True
    except KafkaError as e:
        print(f"Failed to connect to Kafka: {e}")
        return False


def publish_messages(csv_file, topic_name, bootstrap_servers, messages_per_second):
    # Create Kafka producer
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        key_serializer=str.encode,
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
    )

    # Read CSV file
    with open(csv_file, "r") as file:
        csv_reader = csv.DictReader(file)

        for row in csv_reader:
            try:
                # Validate and create TransactionModel instance
                transaction = TransactionModel(**row)

                # Extract credit card number to use as key
                cc_num = transaction.cc_num

                # Convert validated model to dict
                message = transaction.dict(by_alias=True)

                # Send message to Kafka topic with cc_num as key
                producer.send(topic_name, key=cc_num, value=message)

                # Print message for debugging
                print(f"Published: {json.dumps(message, default=str)}")

                # Control publish rate
                time.sleep(1 / messages_per_second)

            except ValidationError as e:
                print(f"Validation error: {e}")
                continue

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
        default="localhost:19092",
        help="Kafka bootstrap servers (default: localhost:19092)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=1.0,
        help="Publish rate in messages per second (default: 1.0)",
    )

    args = parser.parse_args()

    if not check_kafka_connection(args.bootstrap_servers):
        print("Exiting due to Kafka connection failure")
        exit

    # Call the publish_messages function with parsed arguments
    publish_messages(
        args.csv_file, args.topic_name, [args.bootstrap_servers], args.rate
    )
