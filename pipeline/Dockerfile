FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3-pip \
    exiftool \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Symlink python
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && ln -sf /usr/bin/python3 /usr/bin/python

# Install Poetry
RUN pip install poetry==2.4.1

WORKDIR /app

# Install PyTorch with CUDA 12.6 FIRST (before poetry install to avoid CPU override)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Copy project files and install remaining deps via Poetry (no-root installs deps only)
COPY pyproject.toml poetry.lock* ./
COPY src/ src/
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-root --without dev

# Install the package itself
RUN pip install -e . --no-deps

# Entrypoint
ENTRYPOINT ["python", "-m", "photo_ai"]
CMD ["serve"]
