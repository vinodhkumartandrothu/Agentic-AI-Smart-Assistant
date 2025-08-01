ARG PYTHON_VERSION=3.13-slim

FROM python:${PYTHON_VERSION}

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DJANGO_SETTINGS_MODULE=core.settings

RUN mkdir -p /code
WORKDIR /code

COPY requirements.txt /tmp/requirements.txt
RUN set -ex && \
    pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt && \
    rm -rf /root/.cache/

COPY . /code

# Make start.sh executable
RUN chmod +x /code/start.sh

CMD ["./start.sh"]






# ARG PYTHON_VERSION=3.13-slim

# FROM python:${PYTHON_VERSION}

# ENV PYTHONDONTWRITEBYTECODE 1
# ENV PYTHONUNBUFFERED 1
# ENV DJANGO_SETTINGS_MODULE=core.settings

# RUN mkdir -p /code
# WORKDIR /code

# COPY requirements.txt /tmp/requirements.txt
# RUN set -ex && \
#     pip install --upgrade pip && \
#     pip install -r /tmp/requirements.txt && \
#     rm -rf /root/.cache/

# COPY . /code

# RUN python manage.py collectstatic --noinput

# EXPOSE 8000
# CMD ["gunicorn", "--bind", ":8000", "--workers", "2", "core.wsgi"]

