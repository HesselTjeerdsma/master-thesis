from faststream.kafka import KafkaBroker, KafkaMessage
from faststream import FastStream

broker = KafkaBroker("localhost:29092")
app = FastStream(broker)


@broker.subscriber("test-topic")
async def process_message(msg: KafkaMessage):
    print(f"Received message: {msg.content}")
