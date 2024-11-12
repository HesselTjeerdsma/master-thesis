# Import necessary modules and classes
from faststream import FastStream, Logger, ContextRepo, Context
from faststream.confluent import KafkaBroker
from pydantic import ValidationError
from datetime import datetime, date
from decimal import Decimal
import json
import sys
from typing import Union, List, Dict, Any, Optional
from pprint import pprint
import time
from dataclasses import dataclass
from threading import Lock
from pydantic import ValidationError

# Add the parent directory to the Python path to allow importing from sibling directories
sys.path.append("../")

# Import custom modules and classes
from models.transaction import TransactionModel
from models.message import Message
from models.run import Run
from classifiers.fraud_detect import detect_fraud
from models.duck_basemodel import DuckDBModel
from tools.EnergyMeter.energy_meter import EnergyMeter
from tools.SystemInformation.system_information import get_system_config
from langchain_community.llms import LlamaCpp


@dataclass
class ThroughputStats:
    start_time: float
    message_count: int
    last_log_time: float
    lock: Lock


def process_transaction_data(raw_data: dict, logger: Logger) -> dict:
    """
    Process and validate raw transaction data before creating TransactionModel.

    Args:
        raw_data: Dictionary containing transaction data
        logger: Logger instance for debugging

    Returns:
        Processed dictionary with validated data
    """
    processed_data = raw_data.copy()

    # Log the incoming data structure
    logger.debug(
        f"Processing transaction data with keys: {list(processed_data.keys())}"
    )

    try:
        # Convert main transaction fields
        processed_data["amt"] = Decimal(str(processed_data["amt"]))
        processed_data["trans_date_trans_time"] = datetime.strptime(
            processed_data["trans_date_trans_time"], "%Y-%m-%d %H:%M:%S"
        )
        processed_data["dob"] = datetime.strptime(
            processed_data["dob"], "%Y-%m-%d"
        ).date()

        # Ensure numeric fields are correct type
        processed_data["lat"] = float(processed_data["lat"])
        processed_data["long"] = float(processed_data["long"])
        processed_data["city_pop"] = int(processed_data["city_pop"])
        processed_data["merch_lat"] = float(processed_data["merch_lat"])
        processed_data["merch_long"] = float(processed_data["merch_long"])
        processed_data["unix_time"] = int(processed_data["unix_time"])

        # Process recent transactions if present
        if "recent_transactions" in processed_data:
            processed_transactions = []
            for idx, trans in enumerate(processed_data["recent_transactions"]):
                try:
                    processed_trans = trans.copy()
                    logger.debug(
                        f"Processing recent transaction {idx} with keys: {list(processed_trans.keys())}"
                    )

                    # Convert transaction-specific fields
                    processed_trans["timestamp"] = datetime.strptime(
                        processed_trans["timestamp"], "%Y-%m-%d %H:%M:%S"
                    )
                    processed_trans["amount"] = Decimal(str(processed_trans["amount"]))
                    processed_trans["merch_lat"] = float(processed_trans["merch_lat"])
                    processed_trans["merch_long"] = float(processed_trans["merch_long"])
                    if (
                        processed_data["trans_date_trans_time"]
                        < processed_trans["timestamp"]
                    ):
                        print(processed_trans)
                        processed_transactions.append(processed_trans)
                except Exception as e:
                    logger.error(f"Error processing recent transaction {idx}: {str(e)}")
                    raise ValueError(
                        f"Failed to process recent transaction {idx}: {str(e)}"
                    )

            processed_data["recent_transactions"] = processed_transactions

    except Exception as e:
        logger.error(f"Error processing main transaction data: {str(e)}")
        raise ValueError(f"Failed to process transaction data: {str(e)}")

    # Validate all required fields are present
    required_fields = {
        "trans_date_trans_time",
        "cc_num",
        "merchant",
        "category",
        "amt",
        "first",
        "last",
        "gender",
        "street",
        "city",
        "state",
        "zip",
        "lat",
        "long",
        "city_pop",
        "job",
        "dob",
        "trans_num",
        "unix_time",
        "merch_lat",
        "merch_long",
        "is_fraud",
    }

    missing_fields = required_fields - set(processed_data.keys())
    if missing_fields:
        error_msg = f"Missing required fields: {missing_fields}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    return processed_data


# Create a Kafka broker instance
broker = KafkaBroker("localhost:29092")

# Create a FastStream app with the Kafka broker
app = FastStream(broker, title="Transaction Consumer")

# Initialize throughput tracking
stats = ThroughputStats(
    start_time=time.time(), message_count=0, last_log_time=time.time(), lock=Lock()
)

# Initialize the LlamaCpp language model
llm = LlamaCpp(
    model_path="/home/hessel/code/lm-studio/bartowski/Phi-3.5-mini-instruct-GGUF/Phi-3.5-mini-instruct-Q4_K_S.gguf",
    temperature=0.8,
    max_tokens=5000,
    n_ctx=4096,
    n_batch=1024,
    n_gpu_layers=35,
    f16_kv=True,
    verbose=False,
    use_mlock=True,
    use_mmap=False,
    n_threads=6,
)


def calculate_throughput(logger: Logger) -> None:
    """Calculate and log throughput statistics"""
    with stats.lock:
        current_time = time.time()
        elapsed_time = current_time - stats.start_time
        elapsed_since_last_log = current_time - stats.last_log_time

        # Calculate overall throughput
        overall_throughput = (
            stats.message_count / elapsed_time if elapsed_time > 0 else 0
        )

        # Calculate recent throughput (since last log)
        recent_throughput = (
            1 / elapsed_since_last_log if elapsed_since_last_log > 0 else 0
        )

        logger.info(
            f"Throughput Stats:\n"
            f"  Messages Processed: {stats.message_count}\n"
            f"  Overall Throughput: {overall_throughput:.2f} msgs/sec\n"
            f"  Recent Throughput: {recent_throughput:.2f} msgs/sec\n"
            f"  Total Runtime: {elapsed_time:.2f} seconds"
        )

        # Update last log time
        stats.last_log_time = current_time


@broker.subscriber("fraud-detection")
async def consume_transaction(
    msg: Union[str, dict], logger: Logger, run_id: int = Context()
) -> bool:
    """Process incoming transaction messages for fraud detection."""
    # try:
    # Get run_id from context
    if not run_id:
        raise ValueError("Run ID not found in context")

    # Process and validate the message data
    logger.debug(f"Received message type: {type(msg)}")

    if isinstance(msg, str):
        try:
            msg_data = json.loads(msg)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message as JSON: {str(e)}")
            return False
    else:
        msg_data = msg

    # Process the transaction data with detailed logging
    processed_data = process_transaction_data(msg_data, logger)

    # Create transaction model
    logger.debug("Creating TransactionModel with processed data")

    transaction = TransactionModel(**processed_data)

    # Get recent transactions
    print(processed_data)
    recent_transactions = processed_data.get("recent_transactions", [])

    logger.info(
        f"Processing transaction with {len(recent_transactions)} recent transactions "
        f"for card ending in ...{transaction.cc_num[-4:]}"
    )

    # Initialize the EnergyMeter to measure energy consumption
    meter = EnergyMeter(
        disk_avg_speed=1600 * 1e6,
        disk_active_power=6,
        disk_idle_power=1.42,
        include_idle=False,
    )

    # Start energy measurement
    meter.begin()

    # try:
    # Process transaction history into model objects
    processed_history = []
    for hist_trans in recent_transactions:
        if isinstance(hist_trans, dict):
            # Convert datetime fields in history to proper format
            processed_trans = hist_trans.copy()
            for key, value in processed_trans.items():
                if isinstance(value, datetime):
                    processed_trans[key] = value.isoformat()
            # Create TransactionModel from processed dict
            processed_history.append(processed_trans)
        else:
            # If it's already a model, append directly
            processed_history.append(hist_trans)

    logger.debug(f"Transaction model type: {type(transaction)}")
    logger.debug(f"Prepared history entries: {len(processed_history)}")

    # Perform fraud detection with transaction model and processed history
    response = detect_fraud(
        transaction=transaction,  # Keep as TransactionModel
        llm=llm,
        transaction_history=processed_history,
    )

    # End energy measurement
    meter.end()

    # Create and save a Message instance with energy consumption data
    llm_msg = Message.create_llm_message(
        run_id=run_id,
        power_usage=meter.get_total_jules_per_component(),
        prompt=response["prompt"],
        response=response["response"],
        metadata={"fraud": transaction.is_fraud},
    )

    # Update throughput statistics
    with stats.lock:
        stats.message_count += 1

    # Calculate and log throughput
    calculate_throughput(logger)

    logger.info(f"Processed message: {llm_msg.id}")
    return True

    # except Exception as e:
    #    logger.error(f"Error in fraud detection processing: {str(e)}")
    #    raise
    """"
    except Exception as e:
        logger.error(f"Error processing transaction: {str(e)}")
        # Include the full details of the error for debugging
        logger.error(f"Full error details: {type(e).__name__}: {str(e)}")
        return False"""


@app.on_startup
async def setup(logger: Logger, context: ContextRepo) -> None:
    """Initialize the application on startup."""
    try:
        logger.info("Creating Message DBs")
        DuckDBModel.initialize_db(
            "/home/hessel/code/master-thesis/databases/fraud-prod.db"
        )

        run = Run.start(
            model_name="Phi-3.5-mini-instruct-Q4_K_S",
            environment="production",
            metadata=get_system_config(app, llm),
        )

        # Reset throughput statistics
        global stats
        stats = ThroughputStats(
            start_time=time.time(),
            message_count=0,
            last_log_time=time.time(),
            lock=Lock(),
        )

        # Ensure run_id is set in context
        context.set_global("run_id", run.id)
        logger.info(f"Initialized run with ID: {run.id}")
    except Exception as e:
        logger.error(f"Error in setup: {str(e)}")
        raise


@app.after_shutdown
async def setdown(logger: Logger, run_id: int = Context()) -> None:
    """Clean up application resources on shutdown."""
    try:
        if run_id:
            # Log final throughput statistics
            calculate_throughput(logger)

            run = Run.get(run_id)
            run.end()
            logger.info(f"Stopped run with ID: {run_id}")
    except Exception as e:
        logger.error(f"Error in shutdown: {str(e)}")
