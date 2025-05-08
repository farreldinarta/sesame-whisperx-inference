FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python and system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    build-essential \
    git && \
    rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

WORKDIR /code

COPY ./pyproject.toml .

RUN pip3 install --upgrade pip setuptools wheel

RUN pip3 install .

COPY ./app ./app
COPY ./tests ./tests
COPY ./pytest.ini .
COPY ./.env .

ENV PYTHONPATH=/code

ARG ENVIRONMENT=development
ARG CONTAINER_PORT=8000
ENV ENVIRONMENT=$ENVIRONMENT
ENV CONTAINER_PORT=$CONTAINER_PORT

EXPOSE $CONTAINER_PORT

CMD ["sh", "-c", "if [ \"$ENVIRONMENT\" = 'production' ]; then \
        uvicorn app.main:app --host 0.0.0.0 --port ${CONTAINER_PORT}; \
     else \
        uvicorn app.main:app --host 0.0.0.0 --port ${CONTAINER_PORT} --reload; \
     fi"]
