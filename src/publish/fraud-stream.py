import csv
from confluent_kafka import Producer
from confluent_kafka.error import KafkaError
import json
import time
import argparse
import sys
from redis import Redis
from typing import List, Dict, Optional, Deque, Set
from datetime import datetime, timedelta
from collections import deque

sys.path.append("../")

from pydantic import ValidationError
from models.transaction import TransactionModel


class RedisTransactionManager:
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        history_timeframe_days: int = 30,
        expiry_days: int = 30,
    ):
        self.redis = Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True,
        )
        self.history_timeframe_days = history_timeframe_days
        self.expiry_seconds = expiry_days * 24 * 60 * 60
        self.dataset_start_time = None

    def _get_card_key(self, cc_num: str) -> str:
        return f"card_history:{cc_num}"

    def _get_dataset_start_key(self) -> str:
        return "dataset_start_time"

    def set_dataset_start_time(self, start_time: datetime) -> None:
        self.dataset_start_time = start_time
        self.redis.set(
            self._get_dataset_start_key(), start_time.strftime("%Y-%m-%d %H:%M:%S")
        )

    def _get_reference_time(self, transaction_time: datetime) -> datetime:
        if not self.dataset_start_time:
            start_time_str = self.redis.get(self._get_dataset_start_key())
            if start_time_str:
                self.dataset_start_time = datetime.strptime(
                    start_time_str, "%Y-%m-%d %H:%M:%S"
                )

        return transaction_time if self.dataset_start_time else datetime.now()

    def add_transaction(self, transaction: dict) -> dict:
        cc_num = transaction["cc_num"]
        card_key = self._get_card_key(cc_num)

        timestamp = transaction["trans_date_trans_time"]
        if isinstance(timestamp, str):
            trans_datetime = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            timestamp = trans_datetime.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(timestamp, datetime):
            trans_datetime = timestamp
            timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        else:
            raise ValueError("Invalid timestamp format")

        history_entry = {
            "timestamp": timestamp,
            "amount": float(transaction["amt"]),
            "merchant": transaction["merchant"],
            "category": transaction["category"],
            "merch_lat": float(transaction["merch_lat"]),
            "merch_long": float(transaction["merch_long"]),
        }

        pipeline = self.redis.pipeline()
        try:
            pipeline.lpush(card_key, json.dumps(history_entry))
            pipeline.expire(card_key, self.expiry_seconds)
            pipeline.execute()

            history = self.get_transaction_history(
                cc_num, reference_time=trans_datetime
            )[1:]

            enriched_transaction = transaction.copy()
            enriched_transaction["recent_transactions"] = history

            return enriched_transaction

        except Exception as e:
            print(f"Redis error: {e}")
            return transaction

    def get_transaction_history(
        self, cc_num: str, reference_time: Optional[datetime] = None
    ) -> List[Dict]:
        card_key = self._get_card_key(cc_num)
        all_transactions = self.redis.lrange(card_key, 0, -1)

        if reference_time is None:
            reference_time = self._get_reference_time(datetime.now())

        cutoff_date = reference_time - timedelta(days=self.history_timeframe_days)
        filtered_transactions = []

        for t in all_transactions:
            trans_data = json.loads(t)
            trans_date = datetime.strptime(trans_data["timestamp"], "%Y-%m-%d %H:%M:%S")

            if trans_date >= cutoff_date and trans_date <= reference_time:
                filtered_transactions.append(trans_data)

        return filtered_transactions


class TransactionSelector:
    def __init__(self, max_cards: int, selected_cards: set):
        self.max_cards = max_cards
        self.selected_cards: Set[str] = selected_cards
        self.transactions: Dict[str, List[dict]] = {}

    def can_add_card(self, cc_num: str) -> bool:
        return (
            len(self.selected_cards) < self.max_cards or cc_num in self.selected_cards
        )

    def add_transaction(self, transaction: dict) -> None:
        cc_num = transaction["cc_num"]

        if not self.can_add_card(cc_num):
            return

        self.selected_cards.add(cc_num)

        if cc_num not in self.transactions:
            self.transactions[cc_num] = []
        self.transactions[cc_num].append(transaction)

    def get_transaction_count(self) -> int:
        return sum(len(trans) for trans in self.transactions.values())

    def get_fraud_count(self) -> int:
        fraud_count = 0
        for transactions in self.transactions.values():
            fraud_count += sum(1 for trans in transactions if trans["is_fraud"])
        return fraud_count


def delivery_report(err, msg):
    if err is not None:
        print(f"Message delivery failed: {err}")
    else:
        pass


def check_kafka_connection(bootstrap_servers):
    try:
        conf = {"bootstrap.servers": bootstrap_servers}
        producer = Producer(conf)
        producer.flush()
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
    bootstrap_servers: str,
    messages_per_second: float,
    redis_host: str,
    redis_port: int,
    max_cards: int,
    history_timeframe_days: int,
    simulation_timeframe_days: int,
):
    redis_manager = RedisTransactionManager(
        redis_host=redis_host,
        redis_port=redis_port,
        history_timeframe_days=history_timeframe_days,
    )

    transaction_selector = TransactionSelector(
        max_cards,
        selected_cards={
            "30074693890476",
            "4683520018489354",
            "3517527805128735",
            "4497451418073897078",
        },
    )

    conf = {"bootstrap.servers": bootstrap_servers, "on_delivery": delivery_report}
    producer = Producer(conf)

    print("Analyzing transaction date range...")
    min_date = None
    max_date = None

    with open(csv_file, "r") as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            try:
                trans_date_str = row["trans_date_trans_time"]
                trans_date = datetime.strptime(trans_date_str, "%Y-%m-%d %H:%M:%S")

                if min_date is None or trans_date < min_date:
                    min_date = trans_date
                if max_date is None or trans_date > max_date:
                    max_date = trans_date
            except Exception as e:
                continue

    if min_date is None or max_date is None:
        print("No valid transactions found in the file. Exiting.")
        return

    print(f"Transaction date range: {min_date} to {max_date}")

    simulation_start = min_date + timedelta(days=(history_timeframe_days))
    simulation_end = simulation_start + timedelta(days=(simulation_timeframe_days))

    print(f"Using history period: {min_date} to {simulation_start}")
    print(f"Using simulation period: {simulation_start} to {simulation_end}")

    transactions_to_publish = []
    with open(csv_file, "r") as file:
        csv_reader = csv.DictReader(file)
        fraud_c = 0
        legit_c = 0
        for row in csv_reader:
            try:
                transaction = TransactionModel(**row)
                transaction_dict = transaction.dict(by_alias=True)

                trans_date_str = transaction_dict["trans_date_trans_time"]
                if isinstance(trans_date_str, datetime):
                    trans_date = trans_date_str
                else:
                    trans_date = datetime.strptime(trans_date_str, "%Y-%m-%d %H:%M:%S")

                cc_num = transaction_dict["cc_num"]

                if trans_date <= simulation_end:
                    if trans_date < simulation_start:
                        if transaction_selector.can_add_card(cc_num):
                            redis_manager.add_transaction(transaction_dict)
                    else:
                        if transaction_selector.can_add_card(cc_num):
                            if transaction_dict["is_fraud"]:
                                fraud_c += 1
                            else:
                                legit_c += 1
                            transaction_selector.add_transaction(transaction_dict)
                            transactions_to_publish.append(transaction_dict)
                else:
                    print(
                        f"Reached transaction after {simulation_end}, stopping processing."
                    )
                    break

            except ValidationError as e:
                print(f"Validation error: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error: {e}")
                continue

    total_transactions = transaction_selector.get_transaction_count()
    total_fraud = transaction_selector.get_fraud_count()

    if total_transactions == 0 or total_fraud == 0:
        print(
            "No transactions or fraudulent transactions found within specified timeframe. Exiting."
        )
        return

    print(f"\nTransaction Statistics:")
    print(f"Selected cards: {len(transaction_selector.selected_cards)}")
    print(f"Total transactions: {total_transactions}")
    print(f"Total Fraud: {total_fraud}")

    published = 0
    pub_fraud = 0
    last_status_time = time.time()
    status_interval = 5

    all_transactions = []
    for transactions in transaction_selector.transactions.values():
        all_transactions.extend(transactions)
    all_transactions.sort(key=lambda x: x["trans_date_trans_time"])

    print(f"\nProcessing Statistics:")
    print(f"Total transactions to publish: {len(all_transactions)}")

    for transaction in all_transactions:
        try:
            enriched_message = redis_manager.add_transaction(transaction)
            producer.produce(
                topic_name,
                key=transaction["cc_num"],
                value=json.dumps(enriched_message, default=str),
                callback=delivery_report,
            )

            if transaction["is_fraud"]:
                pub_fraud += 1

            published += 1

            current_time = time.time()
            if current_time - last_status_time >= status_interval:
                print(f"\nProgress: Published {published} messages")
                last_status_time = current_time

            time.sleep(1 / messages_per_second)
            producer.poll(0)

        except Exception as e:
            print(f"Unexpected error during publishing: {e}")
            continue

    print("\nFinal Statistics:")
    print(f"Total transactions published: {published} fraud: {pub_fraud}")
    print(f"Unique cards processed: {len(transaction_selector.selected_cards)}")

    producer.flush()


if __name__ == "__main__":
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
        "--redis-host", default="localhost", help="Redis host (default: localhost)"
    )
    parser.add_argument(
        "--redis-port", type=int, default=6379, help="Redis port (default: 6379)"
    )
    parser.add_argument(
        "--max-cards",
        type=int,
        required=True,
        help="Maximum number of unique credit cards to include",
    )
    parser.add_argument(
        "--history-timeframe",
        type=int,
        default=60,
        help="Number of days to keep in transaction history (default: 60)",
    )
    parser.add_argument(
        "--simulation-timeframe",
        type=int,
        default=90,
        help="Number of days to simulate transactions for (default: 90)",
    )

    args = parser.parse_args()

    if args.max_cards < 1:
        print("Error: Maximum number of cards must be at least 1")
        sys.exit(1)

    services_ok = True

    if not check_kafka_connection(args.bootstrap_servers):
        print("Exiting due to Kafka connection failure")
        services_ok = False

    if not check_redis_connection(args.redis_host, args.redis_port):
        print("Exiting due to Redis connection failure")
        services_ok = False

    if not services_ok:
        sys.exit(1)

    print(f"Running with the following parameters:")
    print(f"- Maximum unique cards: {args.max_cards}")
    print(f"- History timeframe: {args.history_timeframe} days")
    print(f"- Simulation timeframe: {args.simulation_timeframe} days")

    publish_messages(
        args.csv_file,
        args.topic_name,
        args.bootstrap_servers,
        args.rate,
        args.redis_host,
        args.redis_port,
        args.max_cards,
        args.history_timeframe,
        args.simulation_timeframe,
    )
