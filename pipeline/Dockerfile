FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# python3.12 is not in Ubuntu 22.04 default repos — deadsnakes PPA required
RUN apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
        exiftool \
        libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m ensurepip --upgrade \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/local/bin/pip3.12 /usr/local/bin/pip

RUN pip install poetry==2.4.1

WORKDIR /app

# Install PyTorch with CUDA 12.6 before poetry to avoid CPU wheel override
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

COPY pyproject.toml poetry.lock* README.md ./
COPY src/ src/
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction

ENTRYPOINT ["python", "-m", "photo_ai"]
CMD ["serve"]
