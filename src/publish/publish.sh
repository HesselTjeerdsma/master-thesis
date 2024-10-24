docker compose -f /home/hessel/code/master-thesis/docker-compose.yml down -v
docker compose -f /home/hessel/code/master-thesis/docker-compose.yml up -d
echo "Waiting untill kafka is avaialable"
sleep 10
python3 fraud-stream.py ~/code/fraudTest.csv --bootstrap-servers localhost:29092