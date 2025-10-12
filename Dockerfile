# Use PyTorch CPU base to avoid building torch from source in CI
FROM pytorch/pytorch:2.0.1-cpu

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy trimmed requirements (no torch/torchvision)
COPY requirements.no-torch.txt /app/requirements.no-torch.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.no-torch.txt

# Copy app code
COPY . /app/

# Create uploads dir
RUN mkdir -p /app/uploads

# Expose port
EXPOSE 5000

# Start the app with gunicorn, use PORT env var if provided by platform
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "app:app"]
