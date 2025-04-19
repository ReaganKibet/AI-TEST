# Use an official Python runtime as a parent image 
FROM python:3.10-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies needed for gevent
RUN apt-get update && apt-get install -y \
    gcc \
    libc-dev \
    python3-dev

# Copy dependency file first, then install dependencies
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . .

# Create a simple mock implementation for the missing modules
RUN mkdir -p /usr/local/lib/python3.10/site-packages/openfabric_pysdk/fields \
    && mkdir -p /usr/local/lib/python3.10/site-packages/openfabric_pysdk/utility \
    && mkdir -p /usr/local/lib/python3.10/site-packages/openfabric_pysdk/helper/proxy \
    && echo "class Resource: pass" > /usr/local/lib/python3.10/site-packages/openfabric_pysdk/fields/__init__.py \
    && echo "class SchemaUtil:\n    @staticmethod\n    def create(obj, data):\n        for key, value in data.items():\n            if hasattr(obj, key):\n                setattr(obj, key, value)\n        return obj" > /usr/local/lib/python3.10/site-packages/openfabric_pysdk/utility/__init__.py \
    && echo "class AppModel: pass\nclass State: pass" >> /usr/local/lib/python3.10/site-packages/openfabric_pysdk/context.py \
    && echo "from .proxy import Proxy, ExecutionResult" > /usr/local/lib/python3.10/site-packages/openfabric_pysdk/helper/__init__.py \
    && echo "class Proxy: pass\nclass ExecutionResult: pass" > /usr/local/lib/python3.10/site-packages/openfabric_pysdk/helper/proxy/__init__.py

# Fix relative import in pipeline_manager.py
RUN sed -i 's/from .llm_manager/from llm_manager/g' /app/app/pipeline_manager.py

# Add this line before the CMD instruction
ENV PYTHONPATH=/app

# Copy the config directory from the correct location
COPY app/config/ config/

# Run the app
CMD ["python", "app/main.py"]