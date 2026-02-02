FROM python:3.13-slim
WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-cache
RUN uv pip install https://github.com/explosion/spacy-models/releases/download/xx_ent_wiki_sm-3.8.0/xx_ent_wiki_sm-3.8.0-py3-none-any.whl

COPY . .

RUN find .venv -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find .venv -name "*.pyc" -delete

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"

CMD ["python", "src/main.py"]
