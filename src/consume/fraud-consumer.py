# Import necessary modules and classes
from faststream import FastStream, Logger, ContextRepo, Context
from faststream.confluent import KafkaBroker
from pydantic import ValidationError
from datetime import datetime, date
from decimal import Decimal
import json
import sys
from typing import Union, List, Dict, Any
from pprint import pprint
import time
from dataclasses import dataclass
from threading import Lock

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
    temperature=0.1,
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


def calculate_throughput(logger: Logger):
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
):
    # Get run_id from context
    if not run_id:
        raise ValueError("Run ID not found in context")

    transaction = TransactionModel(**msg)

    # Get recent transactions directly from the enriched message
    recent_transactions = msg.get("recent_transactions", [])

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

    # Perform fraud detection with transaction history
    response = detect_fraud(
        transaction=transaction,
        llm=llm,
        transaction_history=recent_transactions,
    )

    # End energy measurement
    meter.end()

    # Create and save a Message instance with energy consumption data
    llm_msg = Message.create_llm_message(
        run_id=run_id,
        power_usage=meter.get_total_jules_per_component(),
        prompt=response["prompt"],
        response=response["response"],
    )

    # Update throughput statistics
    with stats.lock:
        stats.message_count += 1

    # Calculate and log throughput
    calculate_throughput(logger)

    logger.info(f"Processed message: {llm_msg.id}")
    return True


@app.on_startup
def setup(logger: Logger, context: ContextRepo):
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
        logger.error(f"Error in setup: {e}")
        raise


@app.after_shutdown
async def setdown(logger: Logger, run_id: int = Context()):
    try:
        if run_id:
            # Log final throughput statistics
            calculate_throughput(logger)

            run = Run.get(run_id)
            run.end()
            logger.info(f"Stopped run with ID: {run_id}")
    except Exception as e:
        logger.error(f"Error in shutdown: {e}")
