import asyncio
import logging
import os
import asyncpg
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from tqdm import tqdm

from config import DATABASE_URL, EMBEDDING_MODEL_NAME, BATCH_SIZE
from clustering import process_item_logic

logging.getLogger("transformers").setLevel(logging.WARNING)

model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(float)
    return np.sum(token_embeddings * input_mask_expanded, axis=1) / np.clip(
        np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None
    )


def encode_texts(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if model is None or tokenizer is None:
        raise RuntimeError("Model not initialized")

    encoded_input = tokenizer(
        texts, padding=True, truncation=True, return_tensors="pt", max_length=512
    ).to(device)

    with torch.no_grad():
        model_output = model(**encoded_input, output_hidden_states=True)
    
    # Move to CPU for numpy
    last_hidden = model_output.last_hidden_state.cpu().numpy()
    attention_mask = encoded_input["attention_mask"].cpu().numpy()
    
    embeddings = mean_pooling(last_hidden, attention_mask)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    final_embeddings = embeddings / norms

    l6_hidden = model_output.hidden_states[6].cpu().numpy()
    l6_embeddings = mean_pooling(l6_hidden, attention_mask)
    l6_norms = np.linalg.norm(l6_embeddings, axis=1, keepdims=True)
    l6_final = l6_embeddings / l6_norms

    return final_embeddings, l6_final


async def run_backfill():
    global model, tokenizer

    print(f"Connecting to {DATABASE_URL}...")
    pool = await asyncpg.create_pool(DATABASE_URL)

    print(f"Loading model {EMBEDDING_MODEL_NAME} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME).to(device)
    model.eval()

    try:
        async with pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT count(*) FROM items WHERE l6_embedding IS NULL"
            )
            print(f"Found {count} items needing L6 embeddings.")

            if count == 0:
                return

            pbar = tqdm(total=count)

            while True:
                rows = await conn.fetch(
                    "SELECT id FROM items WHERE l6_embedding IS NULL LIMIT $1", BATCH_SIZE
                )

                if not rows:
                    break

                for row in rows:
                    try:
                        await process_item_logic(conn, encode_texts, str(row["id"]))
                    except Exception as e:
                        print(f"Error processing {row['id']}: {e}")
                        import traceback
                        traceback.print_exc()

                    pbar.update(1)

    finally:
        await pool.close()
        print("\nDone.")


if __name__ == "__main__":
    asyncio.run(run_backfill())
