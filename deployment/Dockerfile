FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and artifacts
COPY src/ ./src/
COPY deployment/ ./deployment/
COPY models/ ./models/
COPY params.yaml .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "8000"]
