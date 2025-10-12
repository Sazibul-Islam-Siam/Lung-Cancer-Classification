# Use official PyTorch CPU image to avoid installing torch from pip during build
FROM pytorch/pytorch:2.0.1-cpu

# Set environment
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

# Copy requirements and install
# Copy a trimmed requirements file (we exclude torch/torchvision because the base image
# already includes compatible builds). This speeds up builds on Render and avoids
# heavy wheel compilation.
COPY requirements.no-torch.txt /app/requirements.no-torch.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.no-torch.txt

# Copy app code
COPY . /app/

# Create uploads dir
RUN mkdir -p /app/uploads

# Expose port
EXPOSE 5000

# Start the app with gunicorn
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "app:app"]
