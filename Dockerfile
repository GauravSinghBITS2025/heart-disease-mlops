
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python dependencies separately for better layer caching.
COPY requirements-api.txt .
RUN python -m pip install --no-cache-dir --default-timeout=300 --retries=10 -r requirements-api.txt

# Copy application source and the trained model.
COPY src/ ./src/
COPY artifacts/models/final_model.joblib ./artifacts/models/final_model.joblib

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]