# Import necessary modules and classes
from faststream import FastStream, Logger, ContextRepo
from faststream.confluent import KafkaBroker

# FastStream is a specific library for stream processing, used here with Kafka

from pydantic import ValidationError
from datetime import datetime, date
from decimal import Decimal
import json
import sys
from typing import Union

# Add the parent directory to the Python path to allow importing from sibling directories
sys.path.append("../")

# Import custom modules and classes
from models.transaction import TransactionModel
from models.message import Message
from models.run import Run
from classifiers.fraud_detect import detect_fraud

# EnergyMeter is a custom implementation part of a specific library for measuring energy consumption
from tools.EnergyMeter.energy_meter import EnergyMeter


from langchain_community.llms import LlamaCpp


# Create a Kafka broker instance
broker = KafkaBroker("localhost:29092")

# Create a FastStream app with the Kafka broker
app = FastStream(broker, title="Transaction Consumer")

# Initialize the LlamaCpp language model
# Note: This is using a local file path for development purposes
llm = LlamaCpp(
    model_path="/home/hessel/code/lm-studio/bartowski/Phi-3.5-mini-instruct-GGUF/Phi-3.5-mini-instruct-Q4_K_S.gguf",
    temperature=0.1,
    max_tokens=2000,
    n_ctx=2048,
    n_batch=512,
    n_gpu_layers=-1,
    f16_kv=True,
    verbose=False,
    use_mlock=False,
    use_mmap=True,
)


# Startup function to initialize the Message DuckDB database
@app.on_startup
def setup(logger: Logger, context: ContextRepo):
    logger.info("Creating Message DBs")
    Message.initialize_db("../../databases/fraud.db")
    logger.info("Creating Message DBs")
    Run.initialize_db("../../databases/fraud.db")


# Custom JSON decoder to handle datetime, date, and Decimal types
def custom_json_decoder(dct):
    for key, value in dct.items():
        if key == "trans_date_trans_time":
            dct[key] = datetime.fromisoformat(value)
        elif key == "dob":
            dct[key] = date.fromisoformat(value)
        elif key == "amt":
            dct[key] = Decimal(value)
    return dct


# Kafka consumer function
@broker.subscriber("fraud-detection")
async def consume_transaction(msg: Union[str, dict], logger: Logger):
    try:
        # Handle both string and dictionary inputs
        if isinstance(msg, str):
            data = json.loads(msg, object_hook=custom_json_decoder)
        elif isinstance(msg, dict):
            data = custom_json_decoder(msg)
        else:
            raise ValueError(f"Unexpected input type: {type(msg)}")

        # Create a TransactionModel instance from the decoded data
        transaction = TransactionModel(**data)

        # Initialize the EnergyMeter to measure energy consumption
        meter = EnergyMeter(
            disk_avg_speed=1600 * 1e6,
            disk_active_power=6,
            disk_idle_power=1.42,
            include_idle=False,
        )

        # Start energy measurement
        meter.begin()

        # Perform fraud detection
        # detect_fraud uses a language model (llm) to analyze the transaction for potential fraud
        detect_fraud(transaction=transaction, llm=llm)

        # End energy measurement
        meter.end()

        # Create and save a Message instance with energy consumption data
        msg = Message.from_dict(
            meter.get_total_jules_per_component(), run_id="10", message_id="3223"
        )
        msg.save()

        # Log the energy consumption data
        logger.info(meter.get_total_jules_per_component())

        # Log the received transaction
        logger.info(f"Received transaction: {transaction.trans_num}")
        return True

    except ValidationError as e:
        logger.error(f"Validation error: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
