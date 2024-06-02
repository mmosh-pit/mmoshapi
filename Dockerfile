# Use the official Python 3.12 image
FROM python:3.12-slim

# Install system-level dependencies
RUN apt-get update && \
    apt-get install -y gcc python3-dev libopenblas-dev libomp-dev && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy your application code into the container
COPY . .

# Expose the port your application will run on
EXPOSE 8080

# Start your FastAPI application with Uvicorn
# Notice the use of "exec" and shell format to allow environment variable expansion
CMD exec uvicorn app:app --host 0.0.0.0 --port $PORT
