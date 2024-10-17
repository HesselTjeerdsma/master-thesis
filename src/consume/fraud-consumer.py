from faststream import FastStream, Logger
from faststream.confluent import KafkaBroker
from pydantic import ValidationError
from datetime import datetime, date
from decimal import Decimal
import json
import sys
from typing import Union

sys.path.append("../")

from models.transaction import TransactionModel
from classifiers.fraud_detect import detect_fraud
from tools.EnergyMeter.energy_meter import EnergyMeter
from langchain_community.llms import LlamaCpp


import warnings

warnings.filterwarnings("ignore")

# Create a Kafka broker instance
broker = KafkaBroker("localhost:29092")

# Create a FastStream app
app = FastStream(broker, title="Transaction Consumer")
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


# Define a custom JSON decoder to handle datetime and Decimal
def custom_json_decoder(dct):
    for key, value in dct.items():
        if key == "trans_date_trans_time":
            dct[key] = datetime.fromisoformat(value)
        elif key == "dob":
            dct[key] = date.fromisoformat(value)
        elif key == "amt":
            dct[key] = Decimal(value)
    return dct


# Define the consumer function
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

        transaction = TransactionModel(**data)
        meter = EnergyMeter(
            disk_avg_speed=1600 * 1e6,
            disk_active_power=6,
            disk_idle_power=1.42,
            label="Matrix Multiplication",
            include_idle=False,
        )
        meter.begin()
        detect_fraud(transaction=transaction, llm=llm)
        meter.end()
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
