# LineCoach Backend API

LineCoach is a Multi-agent Coaching and Real-time Escalation system designed to enhance customer service interactions. This backend provides the FastAPI endpoints to support the LineCoach functionality.

## Features

* Real-time coaching API for customer service representatives
* Emotion and tone analysis for conversations
* Escalation detection and notification system
* OpenAI integration for advanced language understanding

## Tech Stack

* FastAPI - High-performance web framework
* Pydantic - Data validation and settings management
* Uvicorn - ASGI server implementation
* Google Cloud AI services integration
* OpenAI API integration

## Setup

### Prerequisites

* Python 3.8+
* uv package manager

### Installation

1. Clone the repository
2. Navigate to the backend directory:

```bash
cd backend
```

3. Create a virtual environment and install dependencies using uv:

```bash
# Install uv if you don't have it
pip install uv

# Create a virtual environment
uv venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

4. Set up your environment variables:
   - Copy the `.env.example` file to `.env`
   - Update the values in the `.env` file with your own credentials

## Running the Application

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

The API will be available at http://127.0.0.1:8000

API documentation will be available at:
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

## API Endpoints

- `GET /` - Root endpoint with API information
- `GET /health` - Health check endpoint
- `POST /api/v1/openai/coach` - Get coaching suggestions based on conversation transcript
