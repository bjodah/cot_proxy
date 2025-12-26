# Use official Python base image
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Install required system packages
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project

# Copy application code
ADD . /app

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Expose the application port
EXPOSE 3000

# Set default environment variables
ENV DEBUG=false
ENV LLM_PARAMS=

# Run with Gunicorn
ENTRYPOINT ["uv", "run", "--with", "gunicorn", "gunicorn"]
