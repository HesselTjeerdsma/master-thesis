docker compose -f /home/hessel/code/master-thesis/docker-compose.yml down -v
docker compose -f /home/hessel/code/master-thesis/docker-compose.yml up -d
echo "Waiting untill kafka is avaialable"
sleep 5
python3 fraud-stream.py ~/code/fraudTrain.csv --bootstrap-servers localhost:29092 --redis-host localhost --rate 100000 --max-cards 5 --history-timeframe 45 --simulation-timeframe 15