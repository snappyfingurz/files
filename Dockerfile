# Dockerfile — Self-Improving Customer Support Agent Environment
FROM python:3.11-slim

LABEL maintainer="openenv-example"
LABEL description="Self-Improving Customer Support Agent RL Environment"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY models.py tasks.py grader.py memory.py feedback.py env.py openenv.yaml ./

# Optional: copy example runner if present
COPY example_run.py* ./

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default: run the example script
CMD ["python", "example_run.py"]
