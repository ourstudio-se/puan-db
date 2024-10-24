# Use a slim version of Python
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.6.1 \
    POETRY_HOME="/opt/poetry" \
    PATH="$POETRY_HOME/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

# Set the working directory
WORKDIR /app

# Copy poetry files
COPY poetry.lock pyproject.toml /app/

# Clear Poetry cache before installation
RUN poetry cache clear pypi --all

# Update dependencies and install them
RUN poetry install --no-root --no-dev

# Ensure the virtual environment's bin directory is in PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy the rest of the application code
COPY . /app

# Expose the port the app runs on
EXPOSE 8000

# Set the command to run the application
CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
