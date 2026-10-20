#!/bin/bash
set -e

echo "============================================="
echo "  Voice Biomarker Platform v2 - Starting..."
echo "============================================="

# Wait for Redis to be ready
if [ -n "$REDIS_URL" ]; then
    echo "Waiting for Redis..."
    until python -c "import redis; r = redis.Redis.from_url('$REDIS_URL'); r.ping()" 2>/dev/null; do
        sleep 1
    done
    echo "Redis is ready."
fi

# Ensure directories exist with correct permissions
mkdir -p "$MODEL_PATH" "$DATA_PATH" "$UPLOAD_PATH" /app/logs

# Run database migrations if applicable
# python -m src.db.migrate

echo "Starting application..."
exec "$@"
