import csv
from kafka import KafkaProducer
from kafka.errors import KafkaError
import json
import time
import argparse
import sys
from redis import Redis
from typing import List, Dict, Optional
from datetime import datetime, timedelta

sys.path.append("../")

from pydantic import ValidationError
from models.transaction import TransactionModel


class RedisTransactionManager:
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        history_size: int = 10,
        expiry_days: int = 30,
    ):
        self.redis = Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True,
        )
        self.history_size = history_size
        self.expiry_seconds = expiry_days * 24 * 60 * 60

    def _get_card_key(self, cc_num: str) -> str:
        return f"card_history:{cc_num}"

    def add_transaction(self, transaction: dict) -> dict:
        cc_num = transaction["cc_num"]
        card_key = self._get_card_key(cc_num)

        # Convert datetime to string if it isn't already
        timestamp = transaction["trans_date_trans_time"]
        if isinstance(timestamp, datetime):
            timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")

        # Create history entry with location data
        history_entry = {
            "timestamp": timestamp,
            "amount": float(transaction["amt"]),
            "merchant": transaction["merchant"],
            "category": transaction["category"],
            "merch_lat": float(
                transaction["merch_lat"]
            ),  # Convert to float for consistent JSON handling
            "merch_long": float(transaction["merch_long"]),
        }

        pipeline = self.redis.pipeline()
        try:
            pipeline.lpush(card_key, json.dumps(history_entry))
            pipeline.ltrim(card_key, 0, self.history_size - 1)
            pipeline.expire(card_key, self.expiry_seconds)
            pipeline.execute()

            history = self.get_transaction_history(cc_num)

            enriched_transaction = transaction.copy()
            enriched_transaction["recent_transactions"] = history

            return enriched_transaction

        except Exception as e:
            print(f"Redis error: {e}")
            return transaction

    def get_transaction_history(self, cc_num: str) -> List[Dict]:
        card_key = self._get_card_key(cc_num)
        try:
            transactions = self.redis.lrange(card_key, 0, self.history_size - 1)
            return [json.loads(t) for t in transactions]
        except Exception as e:
            print(f"Error retrieving history: {e}")
            return []


def check_kafka_connection(bootstrap_servers):
    try:
        producer = KafkaProducer(bootstrap_servers=bootstrap_servers)
        producer.close()
        print("Successfully connected to Kafka")
        return True
    except KafkaError as e:
        print(f"Failed to connect to Kafka: {e}")
        return False


def check_redis_connection(host: str, port: int) -> bool:
    try:
        redis = Redis(host=host, port=port)
        redis.ping()
        redis.close()
        print("Successfully connected to Redis")
        return True
    except Exception as e:
        print(f"Failed to connect to Redis: {e}")
        return False


def publish_messages(
    csv_file: str,
    topic_name: str,
    bootstrap_servers: List[str],
    messages_per_second: float,
    redis_host: str,
    redis_port: int,
):
    # Initialize Redis manager
    redis_manager = RedisTransactionManager(
        redis_host=redis_host, redis_port=redis_port, history_size=10, expiry_days=30
    )

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

                # Enrich message with transaction history from Redis
                enriched_message = redis_manager.add_transaction(message)

                # Send enriched message to Kafka topic with cc_num as key
                producer.send(topic_name, key=cc_num, value=enriched_message)

                # Print message for debugging
                print(f"Published enriched message for card ending in ...{cc_num[-4:]}")

                # Control publish rate
                time.sleep(1 / messages_per_second)

            except ValidationError as e:
                print(f"Validation error: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error: {e}")
                continue

    # Ensure all messages are sent
    producer.flush()


if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Publish CSV data to Kafka topic with Redis-enriched transaction history."
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
    parser.add_argument(
        "--redis-host",
        default="localhost",
        help="Redis host (default: localhost)",
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=6379,
        help="Redis port (default: 6379)",
    )

    args = parser.parse_args()

    # Check both Kafka and Redis connections before starting
    services_ok = True

    if not check_kafka_connection(args.bootstrap_servers):
        print("Exiting due to Kafka connection failure")
        services_ok = False

    if not check_redis_connection(args.redis_host, args.redis_port):
        print("Exiting due to Redis connection failure")
        services_ok = False

    if not services_ok:
        sys.exit(1)

    # Call the publish_messages function with parsed arguments
    publish_messages(
        args.csv_file,
        args.topic_name,
        [args.bootstrap_servers],
        args.rate,
        args.redis_host,
        args.redis_port,
    )
