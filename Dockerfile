# Use an official Python runtime as a parent image 
FROM python:3.10-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies needed for gevent and PIL
RUN apt-get update && apt-get install -y \
    gcc \
    libc-dev \
    python3-dev \
    libopenblas-dev \
    libopenjp2-7 \
    libpng-dev \
    libjpeg-dev

# Copy dependency file first, then install dependencies
COPY requirements.txt .

# Install the dependencies with specific attention to compatibility
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "scipy>=1.10.0" && \
    pip install --no-cache-dir pillow && \
    pip install --no-cache-dir flask-apispec && \
    pip install --no-cache-dir "numpy<2.0.0" && \
    pip install --no-cache-dir "torch==2.0.1" --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir sentencepiece && \
    pip install --no-cache-dir "transformers<4.38.0" # Using an older version that's more compatible with CPU-only

# Fix the bitsandbytes issue - install CPU-only version
#RUN pip install --no-cache-dir bitsandbytes-cpu

# Create directories for model offloading and outputs
RUN mkdir -p /app/offload_folder /app/outputs/images /app/logs

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

# Fix relative import in pipeline_manager.py if needed
RUN if [ -f "/app/app/pipeline_manager.py" ]; then \
        sed -i 's/from .llm_manager/from llm_manager/g' /app/app/pipeline_manager.py; \
    fi

# Add this line before the CMD instruction
ENV PYTHONPATH=/app
ENV TRANSFORMERS_OFFLINE=0
ENV HF_HOME=/app/.cache/huggingface
ENV BITSANDBYTES_NOWELCOME=1
ENV OMP_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false

# Copy the config directory from the correct location
COPY app/config/ config/

# Run the app with optimized GC settings for lower memory usage
CMD ["python", "-X", "faulthandler", "app/main.py"]