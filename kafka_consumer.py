from faststream.kafka import KafkaBroker, KafkaMessage
from faststream import FastStream

def create_app():
    broker = KafkaBroker("kafka:9092")
    app = FastStream(broker)

    @broker.subscriber("fraud-detection")
    async def process_fraud_detection(msg: KafkaMessage):
        print(f"Fraud Detection: {msg.value}")

    @broker.subscriber("hatespeech-detection")
    async def process_hatespeech_detection(msg: KafkaMessage):
        print(f"Hate Speech Detection: {msg.value}")

    @broker.subscriber("fakenews-detection")
    async def process_fakenews_detection(msg: KafkaMessage):
        print(f"Fake News Detection: {msg.value}")

    return app

if __name__ == "__main__":
    app = create_app()
    app.run()
