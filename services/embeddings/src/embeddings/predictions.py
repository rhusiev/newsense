import torch
import io
import logging
import asyncpg
import numpy as np
import threading
from .models import UserPreferenceHead

logger = logging.getLogger("predictions")

model_registry: dict[str, tuple[UserPreferenceHead, threading.Lock]] = {}
registry_lock = threading.Lock()


def get_or_create_model(
    user_id: str, input_dim: int = 768
) -> tuple[UserPreferenceHead, threading.Lock]:
    """Retrieves or initializes a model and its lock in the registry."""
    with registry_lock:
        if user_id not in model_registry:
            model = UserPreferenceHead(embedding_dim=input_dim)
            model.eval()
            model_registry[user_id] = (model, threading.Lock())
        return model_registry[user_id]


def update_model_state(user_id: str, model_state_dict: dict):
    """Updates the in-memory model weights safely."""
    input_dim = model_state_dict["network.0.weight"].shape[1]
    model, lock = get_or_create_model(user_id, input_dim)

    with lock:
        model.load_state_dict(model_state_dict)
        model.eval()
    logger.debug(f"In-memory model updated for user {user_id}")


async def load_all_models(conn: asyncpg.Connection):
    """Loads all existing models from the database into the registry on startup."""
    logger.info("Loading all user preference models from database...")
    rows = await conn.fetch("SELECT user_id, model_state FROM user_preference_models")

    count = 0
    for row in rows:
        user_id = str(row["user_id"])
        model_state_bytes = row["model_state"]

        try:
            buffer = io.BytesIO(model_state_bytes)
            state_dict = torch.load(buffer, map_location=torch.device("cpu"))
            update_model_state(user_id, state_dict)
            count += 1
        except Exception as e:
            logger.error(f"Failed to load model for user {user_id}: {e}")

    logger.info(f"Loaded {count} models into memory.")


async def generate_predictions_for_item(
    conn: asyncpg.Connection,
    item_id: str,
    source_id: str,
    embedding: np.ndarray,
    l6_embedding: np.ndarray,
):
    """
    Generates predictions for all relevant users using in-memory models.
    """
    query = """
        SELECT DISTINCT fs.user_id
        FROM feed_subscriptions fs
        JOIN feeds f ON fs.feed_id = f.id
        WHERE f.source_id = $1
    """

    user_rows = await conn.fetch(query, source_id)
    if not user_rows:
        return

    emb_norm = embedding / (np.linalg.norm(embedding) + 1e-9)
    l6_norm = l6_embedding / (np.linalg.norm(l6_embedding) + 1e-9)

    input_768 = torch.FloatTensor(np.concatenate([emb_norm, l6_norm])).unsqueeze(0)
    input_384 = torch.FloatTensor(emb_norm).unsqueeze(0)

    predictions = []

    for row in user_rows:
        user_id = str(row["user_id"])

        with registry_lock:
            if user_id not in model_registry:
                continue
            model, lock = model_registry[user_id]

        try:
            with lock:
                input_dim = next(model.parameters()).shape[1]

                with torch.no_grad():
                    if input_dim == 768:
                        score = model(input_768).item()
                    else:
                        score = model(input_384).item()

            predictions.append((row["user_id"], item_id, float(min(max(score, -1), 1))))
        except Exception as e:
            logger.error(f"Error predicting for user {user_id}: {e}")

    if predictions:
        await conn.executemany(
            """
            INSERT INTO item_predictions (user_id, item_id, score)
            VALUES ($1, $2, $3)
            ON CONFLICT (user_id, item_id) DO UPDATE SET score = EXCLUDED.score
            """,
            predictions,
        )
        logger.info(f"Generated {len(predictions)} predictions for item {item_id}")
