# Start from a slim official Python image matching your local version.
FROM python:3.13-slim

# Don't buffer stdout/stderr (so logs appear immediately) and don't write .pyc.
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies first, in their own layer, so Docker caches them and
# doesn't reinstall on every code change.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy the application code and the model artifacts into the image.
COPY api/ ./api/
COPY artifacts/ ./artifacts/

# Document the port the service listens on.
EXPOSE 8000

# Start the API. Note: no --reload in production, and host 0.0.0.0 so the
# server is reachable from outside the container.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]