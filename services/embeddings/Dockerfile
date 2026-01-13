FROM python:3.13-slim
WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-cache

COPY . .

RUN find .venv -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find .venv -name "*.pyc" -delete

ENV PATH="/app/.venv/bin:$PATH"

CMD ["uv", "run", "python", "main.py"]
