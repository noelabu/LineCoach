# Use the official Python image from the base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_CACHE_DIR=/usr/src/app/.uv_cache

# Install uv (Python package manager)
RUN pip install --no-cache-dir uv

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY pyproject.toml .

# Install dependencies directly into the system environment
RUN uv pip install --system -e .

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8080

# Define environment variable
ENV PORT=8080

# Step 8: Run the FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
