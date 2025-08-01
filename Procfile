web: gunicorn core.wsgi:application
worker: celery -A core.celery worker --loglevel=info
