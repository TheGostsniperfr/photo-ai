FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install poetry==2.4.1

WORKDIR /app

COPY pyproject.toml poetry.lock* ./
COPY src/ src/
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-root --without dev

RUN pip install -e . --no-deps

ENTRYPOINT ["photo-ai-bot"]
