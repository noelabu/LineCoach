# Use the official Python image from the base image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_CACHE_DIR=/usr/src/app/.uv_cache

# Install system dependencies including those needed for PyAudio
RUN apt-get update && apt-get install -y \
    gcc \
    libasound2-dev \
    portaudio19-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv (Python package manager)
RUN pip install --no-cache-dir uv

# Set the working directory in the container
WORKDIR /app

# Copy the entire application code first
COPY . .

# Install dependencies directly into the system environment
RUN uv pip install --system -e .

# Expose the port the app runs on
EXPOSE 8080

# Define environment variable
ENV PORT=8080

# Run the FastAPI app with Uvicorn
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
