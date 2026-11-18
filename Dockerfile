# Aspectra ABSA — Production Container
FROM python:3.12-slim

WORKDIR /app

# Ensure Python output is sent directly to terminal without buffering
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Install lightweight system tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifest
COPY requirements.txt .

# Install PyTorch (CPU-optimized for universal portability on any student/teacher laptop)
# and remaining ML/API dependencies
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code, model checkpoints, data, and pre-compiled Stripe-style frontend
COPY src/ ./src/
COPY api/ ./api/
COPY data/ ./data/
COPY models/ ./models/
COPY results/ ./results/
COPY frontend/dist/ ./frontend/dist/

# Expose unified port
EXPOSE 8000

# Container Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch unified server serving both the REST API and the Stripe-style UI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
