import asyncio
import logging
import os
import asyncpg
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm

from config import DATABASE_URL, EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_PATH, BATCH_SIZE
from clustering import process_item_logic

logging.getLogger("transformers").setLevel(logging.WARNING)

model = None
tokenizer = None


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(float)
    return np.sum(token_embeddings * input_mask_expanded, axis=1) / np.clip(
        np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None
    )


def encode_texts(texts: list[str]) -> np.ndarray:
    if model is None or tokenizer is None:
        raise RuntimeError("Model not initialized")

    encoded_input = tokenizer(
        texts, padding=True, truncation=True, return_tensors="np", max_length=512
    )

    model_output = model(**encoded_input)
    embeddings = mean_pooling(model_output, encoded_input["attention_mask"])

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


async def run_backfill():
    global model, tokenizer

    print(f"Connecting to {DATABASE_URL}...")
    pool = await asyncpg.create_pool(DATABASE_URL)

    print(f"Checking for model at {EMBEDDING_MODEL_PATH}...")

    if os.path.exists(EMBEDDING_MODEL_PATH) and os.path.exists(
        os.path.join(EMBEDDING_MODEL_PATH, "model.onnx")
    ):
        print(f"Loading existing local model from {EMBEDDING_MODEL_PATH}...")
        model = ORTModelForFeatureExtraction.from_pretrained(
            EMBEDDING_MODEL_PATH,
            export=False,
            provider="CPUExecutionProvider",
        )
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)
    else:
        print(f"Model not found locally. Downloading and exporting {EMBEDDING_MODEL_NAME}...")
        model = ORTModelForFeatureExtraction.from_pretrained(
            EMBEDDING_MODEL_NAME,
            export=True,
            provider="CPUExecutionProvider",
        )
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)

        print(f"Saving converted model to {EMBEDDING_MODEL_PATH}...")
        model.save_pretrained(EMBEDDING_MODEL_PATH)
        tokenizer.save_pretrained(EMBEDDING_MODEL_PATH)

    try:
        async with pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT count(*) FROM items WHERE embedding IS NULL"
            )
            print(f"Found {count} items needing embeddings.")

            if count == 0:
                return

            pbar = tqdm(total=count)

            while True:
                rows = await conn.fetch(
                    "SELECT id FROM items WHERE embedding IS NULL LIMIT $1", BATCH_SIZE
                )

                if not rows:
                    break

                for row in rows:
                    try:
                        await process_item_logic(conn, encode_texts, str(row["id"]))
                    except Exception as e:
                        print(f"Error processing {row['id']}: {e}")

                    pbar.update(1)

    finally:
        await pool.close()
        print("\nDone.")


if __name__ == "__main__":
    asyncio.run(run_backfill())
