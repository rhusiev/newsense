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

# Story threads: Leiden communities in a kNN graph of full-text embeddings.
# The chosen production setup is top-10 candidates verified by the storyline LLM.
THREAD_KNN = int(os.getenv("THREAD_KNN", "10"))
THREAD_EMBEDDING_MIN = 0.86
THREAD_RESOLUTION = float(os.getenv("THREAD_RESOLUTION", "2"))
THREAD_GAP_DAYS = 7
# The LLM verifier decides same-storyline pairs; cosine only proposes candidates.
THREAD_LLM_ENABLED = os.getenv("THREAD_LLM_ENABLED", "1").lower() in (
    "1", "true", "yes"
)
THREAD_LLM_THRESHOLD = float(os.getenv("THREAD_LLM_THRESHOLD", "0.7"))
THREAD_LLM_MAX_CALLS = int(os.getenv("THREAD_LLM_MAX_CALLS", "2000"))
THREAD_LLM_DAILY_CALLS = int(os.getenv("THREAD_LLM_DAILY_CALLS", "1000"))
THREAD_LLM_CONCURRENCY = int(os.getenv("THREAD_LLM_CONCURRENCY", "8"))
THREAD_LLM_MAX_CHARS = int(os.getenv("THREAD_LLM_MAX_CHARS", "700"))
# Add neighbors of neighbours so a story chain is not cut by the k limit.
THREAD_TWO_HOP = os.getenv("THREAD_TWO_HOP", "0").lower() in ("1", "true", "yes")
# Optional local pair scorer: a quantized ONNX cross-encoder plus its tokenizer.
THREAD_PAIR_MODEL = os.getenv("THREAD_PAIR_MODEL", "")
THREAD_PAIR_THRESHOLD = float(os.getenv("THREAD_PAIR_THRESHOLD", "0.5"))
THREAD_PAIR_MAX_LEN = int(os.getenv("THREAD_PAIR_MAX_LEN", "128"))
# Inference is batched; a single all-pairs call can allocate gigabytes.
THREAD_PAIR_BATCH = int(os.getenv("THREAD_PAIR_BATCH", "64"))
# A weaker match is shown as an alternative thread rather than joined.
THREAD_ALTERNATIVE_MIN = float(os.getenv("THREAD_ALTERNATIVE_MIN", "0.35"))
# Stories from this many days are grouped together.
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
LLM_SUMMARY_MAX_TOKENS = int(os.getenv("LLM_SUMMARY_MAX_TOKENS", "700"))
LLM_SUMMARY_SETTLE_HOURS = 6
LLM_SUMMARY_MAX_ATTEMPTS = 3
LLM_REQUEST_TIMEOUT_S = 90.0
# Optional JSON object sent as the provider's reasoning field, e.g. {"effort":"low"}.
LLM_REASONING = os.getenv("LLM_REASONING", "").strip()
