FROM ghcr.io/astral-sh/uv:0.13.1 AS uv
FROM python:3.12-slim
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH" DATA_DIR=/app/data/calling_agent
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --only-group calling
COPY calling_agent ./calling_agent
RUN useradd --uid 10001 --create-home mamo && mkdir -p /app/data/calling_agent && chown -R mamo:mamo /app/data
USER mamo
EXPOSE 8080
CMD ["uvicorn", "calling_agent.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--no-access-log"]
