import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgres://user:pass@localhost:5432/authdb")
VALKEY_URL = os.getenv("VALKEY_URL", "redis://localhost:6379")
MODEL_NAME = "intfloat/multilingual-e5-small"

SIMILARITY_THRESHOLD = 0.94
DISTANCE_THRESHOLD = 1 - SIMILARITY_THRESHOLD
TIME_WINDOW_HOURS = 24

QUEUE_NAME = "article_embeddings"
BATCH_SIZE = 50
RECONCILIATION_INTERVAL_MINUTES = int(os.getenv("RECONCILIATION_INTERVAL_MINUTES", "10"))
