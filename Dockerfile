# Use an official Python runtime as a parent image
FROM python:3.12

# Set the working directory in the container
WORKDIR /src

# Install system dependencies including TA-Lib prerequisites
RUN apt-get update && apt-get install -y \
    python3-dev \
    graphviz \
    libgraphviz-dev \
    pkg-config \
    build-essential \
    wget \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# # Install TA-Lib C library (required for ta-lib Python package)
# RUN cd /tmp \
#     && wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz \
#     && tar -xzf ta-lib-0.6.4-src.tar.gz \
#     && cd ta-lib-0.6.4/ \
#     && ./configure --prefix=/usr \
#     && make \
#     && make install \
#     && cd / \
#     && rm -rf /tmp/ta-lib*


# Copy the requirements file and install dependencies
#COPY requirements.txt /src/
#RUN pip install --upgrade pip && pip install -r requirements.txt

# COPY newrequirements.txt /src/
# RUN pip install --upgrade pip && pip install -r newrequirements.txt

# Accept build arguments and set environment variables
ARG SECRET_KEY
ENV SECRET_KEY=${SECRET_KEY}

ARG DEBUG
ENV DEBUG=${DEBUG}

ARG DB_NAME
ENV DB_NAME=${DB_NAME}

ARG DB_USER
ENV DB_USER=${DB_USER}

ARG DB_PASSWORD
ENV DB_PASSWORD=${DB_PASSWORD}

ARG DB_HOST
ENV DB_HOST=${DB_HOST}

ARG DB_PORT_NUMBER
ENV DB_PORT_NUMBER=${DB_PORT_NUMBER}

ARG REDIS_HOST
ENV REDIS_HOST=${REDIS_HOST}

ARG REDIS_PORT_NUMBER
ENV REDIS_PORT_NUMBER=${REDIS_PORT_NUMBER}

ARG REDIS_PASSWORD
ENV REDIS_PASSWORD=${REDIS_PASSWORD}

ARG RABBITMQ_HOST
ENV RABBITMQ_HOST=${RABBITMQ_HOST}

ARG RABBITMQ_PORT_NUMBER
ENV RABBITMQ_PORT_NUMBER=${RABBITMQ_PORT_NUMBER}

ARG RABBITMQ_USER
ENV RABBITMQ_USER=${RABBITMQ_USER}

ARG RABBITMQ_PASSWORD
ENV RABBITMQ_PASSWORD=${RABBITMQ_PASSWORD}

ARG INFLUXDB_URL
ENV INFLUXDB_URL=${INFLUXDB_URL}

ARG INFLUXDB_PORT_NUMBER
ENV INFLUXDB_PORT_NUMBER=${INFLUXDB_PORT_NUMBER}

ARG INFLUXDB_TOKEN
ENV INFLUXDB_TOKEN=${INFLUXDB_TOKEN}

ARG INFLUXDB_ORG
ENV INFLUXDB_ORG=${INFLUXDB_ORG}

ARG INFLUXDB_BUCKET
ENV INFLUXDB_BUCKET=${INFLUXDB_BUCKET}

ARG MONGODB_HOST
ENV MONGODB_HOST=${MONGODB_HOST}

ARG MONGODB_PORT
ENV MONGODB_PORT=${MONGODB_PORT}

ARG MONGODB_USER
ENV MONGODB_USER=${MONGODB_USER} 

ARG MONGODB_PASSWORD
ENV MONGODB_PASSWORD=${MONGODB_PASSWORD}

ARG MONGO_DB_NAME
ENV MONGO_DB_NAME=${MONGO_DB_NAME}

# Copy the entrypoint script first and set execute permissions
COPY entrypoint.sh /src/entrypoint.sh
RUN chmod +x /src/entrypoint.sh

# Copy the source code into the container
COPY src/ /src/

# Create directories for AI models and logs
RUN mkdir -p /src/models /src/logs

# Expose the application port
EXPOSE 8000

# Define the entrypoint
ENTRYPOINT ["./entrypoint.sh"]