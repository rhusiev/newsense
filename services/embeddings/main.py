import asyncio
import logging
import os
from typing import Optional

import asyncpg
from redis import asyncio as aioredis
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
import numpy as np

from config import (
    DATABASE_URL,
    VALKEY_URL,
    MODEL_NAME,
    QUEUE_NAME,
    BATCH_SIZE,
    RECONCILIATION_INTERVAL_MINUTES,
    LOCAL_MODEL_PATH,
)
from clustering import process_item_logic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clustering-service")

db_pool: Optional[asyncpg.Pool] = None
valkey_client: Optional[aioredis.Redis] = None
model: Optional[ORTModelForFeatureExtraction] = None
tokenizer: Optional[AutoTokenizer] = None
shutdown_event = asyncio.Event()


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(float)
    return np.sum(token_embeddings * input_mask_expanded, axis=1) / np.clip(
        np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None
    )


def encode_texts(texts: list[str]) -> np.ndarray:
    encoded_input = tokenizer(
        texts, padding=True, truncation=True, return_tensors="np", max_length=512
    )

    model_output = model(**encoded_input)
    embeddings = mean_pooling(model_output, encoded_input["attention_mask"])

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


async def process_queue_consumer():
    while not shutdown_event.is_set():
        try:
            result = await valkey_client.blpop(QUEUE_NAME, timeout=5)

            if result is None:
                continue

            _, item_id = result
            item_id = item_id.decode("utf-8")

            async with db_pool.acquire() as conn:
                await process_item_logic(conn, encode_texts, item_id)
                logger.info(f"Processed item from queue: {item_id}")

        except Exception as e:
            logger.error(f"Error processing queue item: {e}")
            await asyncio.sleep(1)


async def reconciliation_worker():
    interval_seconds = RECONCILIATION_INTERVAL_MINUTES * 60

    while not shutdown_event.is_set():
        try:
            await asyncio.sleep(interval_seconds)

            async with db_pool.acquire() as conn:
                unprocessed_ids = await conn.fetch(
                    "SELECT id FROM items WHERE embedding IS NULL LIMIT $1", BATCH_SIZE
                )

                if unprocessed_ids:
                    logger.info(
                        f"Reconciliation found {len(unprocessed_ids)} unprocessed items"
                    )

                    for row in unprocessed_ids:
                        await process_item_logic(conn, encode_texts, str(row["id"]))
                        logger.info(f"Reconciliation processed: {row['id']}")

        except Exception as e:
            logger.error(f"Reconciliation error: {e}")


async def startup():
    global db_pool, valkey_client, model, tokenizer

    logger.info(f"Checking for model at {LOCAL_MODEL_PATH}...")

    if os.path.exists(LOCAL_MODEL_PATH) and os.path.exists(
        os.path.join(LOCAL_MODEL_PATH, "model.onnx")
    ):
        logger.info(f"Loading existing local model from {LOCAL_MODEL_PATH}...")
        model = ORTModelForFeatureExtraction.from_pretrained(
            LOCAL_MODEL_PATH,
            export=False,
            provider="CPUExecutionProvider",
        )
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    else:
        logger.info(
            f"Model not found locally. Downloading and exporting {MODEL_NAME}..."
        )
        model = ORTModelForFeatureExtraction.from_pretrained(
            MODEL_NAME,
            export=True,
            provider="CPUExecutionProvider",
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        logger.info(f"Saving converted model to {LOCAL_MODEL_PATH}...")
        model.save_pretrained(LOCAL_MODEL_PATH)
        tokenizer.save_pretrained(LOCAL_MODEL_PATH)

    logger.info("Model loaded")

    logger.info("Connecting to database...")
    db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    logger.info("Database connected")

    logger.info("Connecting to Valkey...")
    valkey_client = await aioredis.from_url(VALKEY_URL, decode_responses=False)
    await valkey_client.ping()
    logger.info("Valkey connected")


async def shutdown():
    logger.info("Shutting down...")
    shutdown_event.set()

    if db_pool:
        await db_pool.close()
    if valkey_client:
        await valkey_client.close()


async def main():
    await startup()

    queue_task = asyncio.create_task(process_queue_consumer())
    reconciliation_task = asyncio.create_task(reconciliation_worker())

    try:
        await asyncio.gather(queue_task, reconciliation_task)
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        await shutdown()


if __name__ == "__main__":
    asyncio.run(main())
