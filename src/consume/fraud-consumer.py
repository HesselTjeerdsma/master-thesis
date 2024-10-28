# Import necessary modules and classes
from faststream import FastStream, Logger, ContextRepo, Context
from faststream.confluent import KafkaBroker
from ksql import KSQLAPI
from pydantic import ValidationError
from datetime import datetime, date
from decimal import Decimal
import json
import sys
from typing import Union, List, Dict, Any

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

# Create a Kafka broker instance
broker = KafkaBroker("localhost:29092")

# Create a FastStream app with the Kafka broker
app = FastStream(broker, title="Transaction Consumer")

# Initialize KSQL client
ksql_client = KSQLAPI("http://localhost:8088")

# Initialize the LlamaCpp language model
llm = LlamaCpp(
    model_path="/home/hessel/code/lm-studio/bartowski/Phi-3.5-mini-instruct-GGUF/Phi-3.5-mini-instruct-Q4_K_S.gguf",
    temperature=0.1,
    max_tokens=2000,
    n_ctx=4096,
    n_batch=1024,
    n_gpu_layers=35,
    f16_kv=True,
    verbose=False,
    use_mlock=True,
    use_mmap=False,
    n_threads=6,
)


async def get_recent_transactions(cc_num: str) -> List[Dict[str, Any]]:
    """Fetch the last 10 transactions for a given credit card number."""
    query = f"""
    SELECT 
        cc_num,
        timestamp,
        amount,
        merchant,
        category,
        event_time
    FROM recent_transactions_stream
    WHERE cc_num = '{cc_num}'
    EMIT CHANGES
    LIMIT 10"""

    try:
        # Execute the query and get the generator
        result_generator = ksql_client.query(query)

        transactions = []

        # Process results from the generator
        try:
            for result in result_generator:
                if isinstance(result, list) and len(result) >= 5:
                    # Parse the list result where indices correspond to SELECT statement order
                    transactions.append(
                        {
                            "timestamp": result[1],  # timestamp is second column
                            "amount": float(result[2]),  # amount is third column
                            "merchant": result[3],  # merchant is fourth column
                            "category": result[4],  # category is fifth column
                        }
                    )

                    if len(transactions) >= 10:
                        break

        except Exception as e:
            print(f"Error processing query result: {e}")
            print(f"Result type: {type(result)}, value: {result}")
            return []
        finally:
            # Clean up the generator
            try:
                result_generator.close()
            except:
                pass

        return transactions

    except Exception as e:
        print(f"Error fetching recent transactions: {e}")
        return []


def create_transaction_stream():
    # Create the base stream from the Kafka topic
    create_stream_query = """
    CREATE STREAM IF NOT EXISTS transactions_stream (
        cc_num VARCHAR,
        trans_date_trans_time VARCHAR,
        amt DOUBLE,
        merchant VARCHAR,
        category VARCHAR
    ) WITH (
        kafka_topic='fraud-detection',
        value_format='JSON',
        key_format='KAFKA'
    )"""

    # Create a stream of recent transactions that we can query
    create_recent_stream_query = """
    CREATE STREAM IF NOT EXISTS recent_transactions_stream AS
    SELECT 
        cc_num,
        trans_date_trans_time AS timestamp,
        CAST(amt AS STRING) AS amount,
        merchant,
        category,
        ROWTIME AS event_time
    FROM transactions_stream
    PARTITION BY cc_num"""

    try:
        # Drop existing streams if they exist
        ksql_client.ksql(
            "DROP STREAM IF EXISTS recent_transactions_stream DELETE TOPIC"
        )
        ksql_client.ksql("DROP STREAM IF EXISTS transactions_stream DELETE TOPIC")

        # Create new streams
        print("Creating transactions stream...")
        ksql_client.ksql(create_stream_query)
        print("Creating recent transactions stream...")
        ksql_client.ksql(create_recent_stream_query)
        print("KSQL streams created successfully")

    except Exception as e:
        print(f"Error creating KSQL streams: {e}")
        raise


@broker.subscriber("fraud-detection")
async def consume_transaction(
    msg: Union[str, dict], logger: Logger, run_id: int = Context()
):

    try:
        # Get run_id from context
        if not run_id:
            raise ValueError("Run ID not found in context")

        transaction = TransactionModel(**msg)

        # Fetch recent transactions for the current credit card
        recent_transactions = await get_recent_transactions(transaction.cc_num)

        print(recent_transactions)
        return True

        logger.info(
            f"Found {len(recent_transactions)} recent transactions for card ending in ...{transaction.cc_num[-4:]}"
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
            llm=llm,  # transaction_history=recent_transactions
        )

        response = {}
        response["prompt"] = "test"
        response["response"] = "response"
        # End energy measurement
        meter.end()

        # Create and save a Message instance with energy consumption data
        llm_msg = Message.create_llm_message(
            run_id=run_id,  # Use run_id from context
            power_usage=meter.get_total_jules_per_component(),
            prompt=response["prompt"],
            response=response["response"],
        )

        #  logger.info(f"Processed message: {llm_msg.id}")
        return True

    except ValidationError as e:
        logger.error(f"Validation error: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
    except ValueError as e:
        logger.error(f"Value error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


# Modified setup function to ensure run_id is properly set in context
@app.on_startup
def setup(logger: Logger, context: ContextRepo):
    try:
        logger.info("Creating Message DBs and KSQL Stream")
        DuckDBModel.initialize_db(
            "/home/hessel/code/master-thesis/databases/fraud-prod.db"
        )
        create_transaction_stream()

        run = Run.start(
            model_name="Phi-3.5-mini-instruct-Q4_K_S",
            environment="production",
            metadata=get_system_config(app, llm),
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
            run = Run.get(run_id)
            run.end()
            logger.info(f"Stopped run with ID: {run_id}")
    except Exception as e:
        logger.error(f"Error in shutdown: {e}")
