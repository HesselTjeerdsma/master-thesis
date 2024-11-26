#!/bin/bash

CONSUMER_PATTERN="/usr/bin/python3 /usr/local/bin/faststream run fraud-consumer:app"
CONSUMER_PID=""

cleanup() {
    echo "$(date): Gracefully shutting down consumer..."
    if [ -n "$CONSUMER_PID" ]; then
        # Send SIGTERM first
        kill "$CONSUMER_PID" 2>/dev/null
        
        # Wait up to 10 seconds for graceful shutdown
        for i in {1..10}; do
            if ! ps -p "$CONSUMER_PID" > /dev/null; then
                break
            fi
            sleep 1
        done
        
        # If process still exists after 10 seconds, then force kill
        if ps -p "$CONSUMER_PID" > /dev/null; then
            echo "$(date): Force killing consumer after 10s timeout..."
            kill -9 "$CONSUMER_PID" 2>/dev/null
        fi
    fi
    
    # Cleanup any remaining processes (backup)
    pkill -TERM -f "$CONSUMER_PATTERN" 2>/dev/null
    exit 0
}

# Handle Ctrl+C (SIGINT)
trap cleanup SIGINT

while true; do
    echo "$(date): Starting faststream consumer"
    /usr/bin/python3 /home/hessel/.local/bin/faststream run fraud-consumer:app > >(tee /tmp/fraud_consumer.log) &
    CONSUMER_PID=$!
    
    LAST_ACTIVITY=$(date +%s)
    LAST_SIZE=0
    
    while true; do
        if ! ps -p $CONSUMER_PID > /dev/null; then
            break
        fi
        
        CURRENT_TIME=$(date +%s)
        CURRENT_SIZE=$(wc -c < /tmp/fraud_consumer.log)
        
        if [ "$CURRENT_SIZE" -gt "$LAST_SIZE" ]; then
            LAST_ACTIVITY=$CURRENT_TIME
            LAST_SIZE=$CURRENT_SIZE
        fi
        
        IDLE_TIME=$((CURRENT_TIME - LAST_ACTIVITY))
        if [ $IDLE_TIME -gt 20 ]; then
            echo "$(date): Consumer inactive for $IDLE_TIME seconds, shutting down..."
            kill -9 $CONSUMER_PID
            
            break
        fi
        
        sleep 5
    done
    
    echo "$(date): Consumer ended, restarting..."
    sleep 1
done
