FROM python:3.12.3-slim as builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libopenblas-dev

RUN pip install poetry==1.8.3

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN poetry install && rm -rf $POETRY_CACHE_DIR

FROM python:3.12.3-slim as runtime

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

COPY . .

EXPOSE 8000

USER 65534

ENTRYPOINT ["fastapi", "run", "main.py", "--port", "8000"]
