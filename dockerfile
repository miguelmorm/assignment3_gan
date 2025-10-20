# --- Dockerfile for GAN FastAPI ---
FROM python:3.10-slim

# Install minimal OS packages and Python dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo-dev zlib1g-dev git && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency list and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY helper_lib ./helper_lib
COPY app ./app
COPY checkpoints ./checkpoints
COPY outputs ./outputs

# Expose FastAPI port
EXPOSE 8000

# Run server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
