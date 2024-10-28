#!/bin/bash

# Color codes for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Connecting to ksqlDB...${NC}"

# Function to execute ksql command
execute_ksql() {
    echo -e "${GREEN}Executing: $1${NC}"
    echo $1 | docker-compose exec -T ksqldb-cli ksql http://ksqldb-server:8088
    echo -e "${BLUE}----------------------------------------${NC}"
    sleep 2  # Give some time between commands
}

# Test basic connectivity
execute_ksql "SHOW STREAMS;"

# Create the transactions stream
execute_ksql "
CREATE STREAM IF NOT EXISTS transactions_stream (
    index BIGINT,
    transaction_id BIGINT,
    transaction_timestamp VARCHAR,
    card_number VARCHAR,
    merchant VARCHAR,
    category VARCHAR,
    amount DOUBLE,
    first_name VARCHAR,
    last_name VARCHAR,
    gender VARCHAR,
    street VARCHAR,
    city VARCHAR,
    state VARCHAR,
    zip VARCHAR,
    lat DOUBLE,
    lon DOUBLE,
    city_pop BIGINT,
    job VARCHAR,
    dob VARCHAR,
    merch_lat DOUBLE,
    merch_lon DOUBLE,
    is_fraud BIGINT
) WITH (
    KAFKA_TOPIC = 'fraud-detection',
    VALUE_FORMAT = 'JSON',
    TIMESTAMP = 'transaction_timestamp',
    TIMESTAMP_FORMAT = 'yyyy-MM-dd HH:mm:ss'
);"

# Verify stream creation
execute_ksql "DESCRIBE transactions_stream;"

# Create a derived stream for suspicious transactions
execute_ksql "
CREATE STREAM IF NOT EXISTS suspicious_transactions AS
SELECT 
    transaction_timestamp,
    card_number,
    merchant,
    category,
    amount,
    first_name,
    last_name,
    city,
    state,
    is_fraud
FROM transactions_stream
WHERE amount > 200 OR is_fraud = 1
EMIT CHANGES;"

# Create a stream for transactions by category
execute_ksql "
CREATE STREAM IF NOT EXISTS category_transactions AS
SELECT 
    category,
    COUNT(*) as transaction_count,
    SUM(amount) as total_amount,
    AVG(amount) as avg_amount
FROM transactions_stream
WINDOW TUMBLING (SIZE 1 HOUR)
GROUP BY category
EMIT CHANGES;"

# Test queries
echo -e "${BLUE}Testing queries...${NC}"

# Show all streams
execute_ksql "SHOW STREAMS;"

# Query recent transactions
execute_ksql "
SELECT 
    transaction_timestamp,
    card_number,
    amount,
    merchant,
    is_fraud
FROM transactions_stream
LIMIT 5;"

# Query suspicious transactions
execute_ksql "
SELECT 
    transaction_timestamp,
    card_number,
    amount,
    merchant,
    is_fraud
FROM suspicious_transactions
LIMIT 5;"

# Query category analytics
execute_ksql "
SELECT 
    category,
    transaction_count,
    total_amount,
    avg_amount
FROM category_transactions
LIMIT 5;"

echo -e "${GREEN}Test script completed!${NC}"