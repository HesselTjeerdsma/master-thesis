sudo docker compose -f ~/code/master-thesis/docker-compose.yml down -v
sudo docker compose -f ~/code/master-thesis/docker-compose.yml up -d
echo "Waiting untill kafka is avaialable"
sleep 5
python3 fraud-stream.py ~/code/fraudTrain.csv --bootstrap-servers localhost:29092 --redis-host localhost --rate 100000 --max-cards 258 --history-timeframe 90 --simulation-timeframe 2