#!/bin/sh

# Exit on error
set -e

# This is the database file the app will create on its first run
DB_FILE="/data/users.db"

# Wait for the database file to exist.
# The `ls` command will fail until the volume is mounted and the app has created the DB.
# We loop until it succeeds.
until ls "$DB_FILE" >/dev/null 2>&1; do
  echo "Waiting for database file to be created by the app..."
  sleep 1
done

echo "Database file found. Starting Gunicorn."
# Start the Gunicorn web server
gunicorn --bind 0.0.0.0:8080 --timeout 120 --workers 1 app:app