import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgres://user:pass@localhost:5432/authdb")
VALKEY_URL = os.getenv("VALKEY_URL", "redis://localhost:6379")
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
LOCAL_MODEL_PATH = "./onnx_model_data"

SIMILARITY_THRESHOLD = 0.89
DISTANCE_THRESHOLD = 1 - SIMILARITY_THRESHOLD
TIME_WINDOW_HOURS = 24

QUEUE_NAME = "article_embeddings"
BATCH_SIZE = 50
RECONCILIATION_INTERVAL_MINUTES = int(os.getenv("RECONCILIATION_INTERVAL_MINUTES", "10"))
