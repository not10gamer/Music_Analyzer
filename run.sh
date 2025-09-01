#!/bin/sh

# Exit on error
set -e

# Start the Gunicorn web server
gunicorn --bind 0.0.0.0:8080 --timeout 120 --workers 1 app:app