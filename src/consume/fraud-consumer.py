# Import necessary modules and classes
from faststream import FastStream, Logger, ContextRepo, Context
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
from models.duck_basemodel import DuckDBModel

# EnergyMeter is a custom implementation part of a specific library for measuring energy consumption
from tools.EnergyMeter.energy_meter import EnergyMeter
from tools.SystemInformation.system_information import get_system_config


from langchain_community.llms import LlamaCpp


# Create a Kafka broker instance
broker = KafkaBroker("localhost:29092")

# Create a FastStream app with the Kafka broker
app = FastStream(broker, title="Transaction Consumer")

# Initialize the LlamaCpp language model
# Optimized settings for RTX 3070 (8GB VRAM)
llm = LlamaCpp(
    model_path="/home/hessel/code/lm-studio/bartowski/Phi-3.5-mini-instruct-GGUF/Phi-3.5-mini-instruct-Q4_K_S.gguf",
    temperature=0.1,
    max_tokens=2000,
    n_ctx=4096,  # Increased context window
    n_batch=1024,  # Increased batch size
    n_gpu_layers=35,  # Specific number of layers to offload
    f16_kv=True,  # Keep half-precision
    verbose=False,  # Temporarily enable for performance monitoring
    use_mlock=True,  # Enable memory locking
    use_mmap=False,  # Disable memory mapping
    n_threads=6,  # Add thread count optimization
)


# Startup function to initialize the Message DuckDB database
@app.on_startup
def setup(logger: Logger, context: ContextRepo):
    logger.info("Creating Message DBs")
    DuckDBModel.initialize_db(
        "/home/hessel/code/master-thesis/databases/fraud-prod.db"
    )  # Use ':memory:' for in-memory database
    run = Run.start(
        model_name="Phi-3.5-mini-instruct-Q4_K_S",
        environment="production",
        metadata=get_system_config(app, llm),
    )

    # Save run ID to FastStream context for later access
    context.set_global("run_id", run.id)
    logger.info(f"Initialized run with ID: {run.id}")


@app.after_shutdown
def setdown(
    logger: Logger,
    run_id: str = Context(),
):
    run = Run.get(run_id)
    run.end()
    logger.info(f"Stoped run with ID: {run_id}")


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
async def consume_transaction(
    msg: Union[str, dict],
    logger: Logger,
    run_id: str = Context(),
):
    try:

        transaction = TransactionModel(**msg)

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
        response = detect_fraud(transaction=transaction, llm=llm)

        # End energy measurement
        meter.end()

        # Get run_id from context
        print(run_id)
        # Create and save a Message instance with energy consumption data

        llm_msg = Message.create_llm_message(
            run_id=run_id,
            power_usage=meter.get_total_jules_per_component(),
            prompt=response["prompt"],
            response=response["response"],
        )

        # Log the received transaction
        logger.info(f"Received message: {llm_msg.id}")
        return True

    except ValidationError as e:
        logger.error(f"Validation error: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
