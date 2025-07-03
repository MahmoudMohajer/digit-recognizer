# Use official Python image
FROM python:3.12.2

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY pyproject.toml .
COPY poetry.lock .
RUN pip install --no-cache-dir poetry
RUN poetry install --only main

# Copy backend, model, and saved_model folders
COPY backend/ ./backend/
COPY saved_model/ ./saved_model/
COPY frontend/ ./frontend

# Expose FastAPI port
EXPOSE 8000

# Set environment variables (optional, e.g., for wandb)
ENV PYTHONUNBUFFERED=1

# Start FastAPI app
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]