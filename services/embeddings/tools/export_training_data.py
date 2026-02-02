import asyncio
import csv
import asyncpg
import argparse
from datetime import datetime
from pathlib import Path
from embeddings.config import DATABASE_URL

EXPORT_DIR = Path("./training_data")


async def export_user_training_data(cutoff_date: datetime | None = None):
    EXPORT_DIR.mkdir(exist_ok=True)
    
    conn = await asyncpg.connect(DATABASE_URL)
    
    try:
        users_with_ratings = await conn.fetch("""
            SELECT DISTINCT user_id 
            FROM item_reads 
            WHERE liked IS NOT NULL
        """)
        
        if not users_with_ratings:
            print("No users with ratings found")
            return
        
        for user_row in users_with_ratings:
            user_id = str(user_row["user_id"])
            
            base_query = """
                SELECT 
                    ir.item_id,
                    ir.liked,
                    i.title,
                    i.content,
                    i.embedding,
                    i.l6_embedding
                FROM item_reads ir
                JOIN items i ON ir.item_id = i.id
                WHERE ir.user_id = $1 
                  AND ir.liked IS NOT NULL
                  AND i.embedding IS NOT NULL
            """
            
            if cutoff_date:
                ratings = await conn.fetch(
                    base_query + " AND i.published_at <= $2 ORDER BY ir.marked_at DESC",
                    user_row["user_id"],
                    cutoff_date
                )
            else:
                ratings = await conn.fetch(
                    base_query + " ORDER BY ir.marked_at DESC",
                    user_row["user_id"]
                )
            
            if not ratings:
                print(f"User {user_id}: No valid ratings found")
                continue
            
            cutoff_suffix = f"_{cutoff_date.strftime('%Y-%m-%d_%H-%M')}" if cutoff_date else ""
            csv_path = EXPORT_DIR / f"user_{user_id}{cutoff_suffix}.csv"
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["item_id", "liked", "title", "content", "embedding", "l6_embedding"])
                
                for row in ratings:
                    writer.writerow([
                        str(row["item_id"]),
                        row["liked"],
                        row["title"] or "",
                        row["content"] or "",
                        row["embedding"],
                        row["l6_embedding"]
                    ])
            
            print(f"User {user_id}: Exported {len(ratings)} ratings to {csv_path}")
    
    finally:
        await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export user training data with optional date cutoff"
    )
    parser.add_argument(
        "--cutoff-date",
        type=str,
        help="Only include items published up to this date (format: YYYY-MM-DD_HH-MM)",
        default=None
    )
    
    args = parser.parse_args()
    
    cutoff_date = None
    if args.cutoff_date:
        try:
            cutoff_date = datetime.strptime(args.cutoff_date, "%Y-%m-%d_%H-%M")
            print(f"Using cutoff date: {cutoff_date}")
        except ValueError:
            print(f"Error: Invalid date format '{args.cutoff_date}'. Use YYYY-MM-DD_HH-MM")
            exit(1)
    
    asyncio.run(export_user_training_data(cutoff_date))
