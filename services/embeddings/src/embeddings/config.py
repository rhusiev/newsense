import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgres://user:pass@localhost:5432/authdb")
VALKEY_URL = os.getenv("VALKEY_URL", "redis://localhost:6379")

# clustering
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
BATCH_SIZE = 50

SIMILARITY_THRESHOLD = 0.94
DISTANCE_THRESHOLD = 1 - SIMILARITY_THRESHOLD
TIME_WINDOW_HOURS = 24

TITLE_WEIGHT = 0.8
CONTENT_WEIGHT = 0.1
ENTITY_WEIGHT = 0.05

QUEUE_NAME = "article_embeddings"
RECONCILIATION_INTERVAL_MINUTES = int(
    os.getenv("RECONCILIATION_INTERVAL_MINUTES", "10")
)
TRAINING_INTERVAL_MINUTES = int(os.getenv("TRAINING_INTERVAL_MINUTES", "30"))

# preference
TRAINING_DATA_DIR = "./training_data"
MODEL_OUTPUT_DIR = "./preference_models"
EMBEDDING_DIM = 384

REPLAY_BUFFER_SIZE = 64
MIN_TRAINING_SAMPLES = 5

# Story threads (replay of 2026 data, 28 days: 0.90 / 0.20 joined 193 threads of 3+ stories).
THREAD_EMBEDDING_MIN = 0.90
THREAD_TITLE_WORDS_MIN = 0.20
THREAD_GAP_DAYS = 7
# A story is compared with this many of the thread's latest stories.
THREAD_RECENT_STORIES = 5
# Rare-word weights come from the titles of this many days.
THREAD_VOCABULARY_DAYS = 28
# A story is threaded once it is this old, so most of its articles have joined it.
THREAD_SETTLE_HOURS = 6
THREADS_INTERVAL_MINUTES = int(os.getenv("THREADS_INTERVAL_MINUTES", "30"))

# LLM story summaries. One OpenAI-compatible endpoint; the worker does nothing
# while the base URL, model or key is empty.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
# Optional JSON provider routing, e.g. {"data_collection":"deny","zdr":true} for OpenRouter.
LLM_PROVIDER_ROUTING = os.getenv("LLM_PROVIDER_ROUTING", "")
LLM_SUMMARY_INTERVAL_MINUTES = int(os.getenv("LLM_SUMMARY_INTERVAL_MINUTES", "5"))
LLM_SUMMARY_BATCH = int(os.getenv("LLM_SUMMARY_BATCH", "5"))
LLM_SUMMARY_MAX_WORDS = 180
LLM_SUMMARY_MAX_CHARS = 12000
LLM_SUMMARY_SETTLE_HOURS = 6
LLM_SUMMARY_MAX_ATTEMPTS = 3
LLM_REQUEST_TIMEOUT_S = 90.0
