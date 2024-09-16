from faststream import FastStream, Logger
from faststream.kafka import KafkaBroker
from pydantic import ValidationError
from datetime import datetime, date
from decimal import Decimal
import json
import sys
from typing import Union

sys.path.append("../")

from models.transaction import TransactionModel

# Create a Kafka broker instance
broker = KafkaBroker("localhost:29092")

# Create a FastStream app
app = FastStream(broker, title="Transaction Consumer")


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

        # Log the received transaction
        logger.info(f"Received transaction: {transaction.trans_num}")

    except ValidationError as e:
        logger.error(f"Validation error: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
