FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

RUN python -m pip install --upgrade pip \
    && python -m pip install .

RUN mkdir -p /app/data /app/artifacts \
    && chown -R app:app /app

USER app

EXPOSE 8000

CMD ["uvicorn", "rssi_localization.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS test

USER root

COPY tests ./tests

RUN python -m pip install ".[dev]" \
    && chown -R app:app /app

USER app

CMD ["pytest"]
