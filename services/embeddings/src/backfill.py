import asyncio
import logging
import os
import asyncpg
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from tqdm import tqdm

from embeddings.config import DATABASE_URL, EMBEDDING_MODEL_NAME, BATCH_SIZE
from embeddings.clustering import process_item_logic
from embeddings.utils import encode_texts
from embeddings.training import train_user_preference_model, get_users_needing_training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backfill")
logging.getLogger("transformers").setLevel(logging.WARNING)

model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_texts_wrapper(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    return encode_texts(texts, model, tokenizer, device)


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
            # 1. Backfill Item Embeddings
            count = await conn.fetchval(
                "SELECT count(*) FROM items WHERE l6_embedding IS NULL"
            )
            
            if count > 0:
                print(f"Found {count} items needing L6 embeddings.")
                pbar = tqdm(total=count, desc="Items")

                while True:
                    rows = await conn.fetch(
                        "SELECT id FROM items WHERE l6_embedding IS NULL LIMIT $1", BATCH_SIZE
                    )

                    if not rows:
                        break

                    for row in rows:
                        try:
                            await process_item_logic(conn, encode_texts_wrapper, str(row["id"]))
                        except Exception as e:
                            print(f"Error processing {row['id']}: {e}")

                        pbar.update(1)
                pbar.close()
            else:
                print("No items needing L6 embeddings.")

            # 2. Train User Preference Models
            print("\nChecking for users needing model training...")
            users_to_train = await get_users_needing_training(conn, force_all=True)
            
            if users_to_train:
                print(f"Found {len(users_to_train)} users to train.")
                pbar = tqdm(total=len(users_to_train), desc="Users")
                for row in users_to_train:
                    await train_user_preference_model(conn, row["user_id"], row["latest_activity"])
                    pbar.update(1)
                pbar.close()
            else:
                print("No users to train.")

    finally:
        await pool.close()
        print("\nDone.")


if __name__ == "__main__":
    asyncio.run(run_backfill())
