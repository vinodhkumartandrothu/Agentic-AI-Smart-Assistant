#!/bin/bash

# Activate your virtual environment if needed
source /venv/bin/activate


# Start the Celery worker
celery -A core worker --loglevel=info 


