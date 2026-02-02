import asyncio
import os
import json
import asyncpg
from datetime import datetime
from embeddings.config import DATABASE_URL

# --- Config ---
SHOW_INPUT_TEXT = True 

# ANSI Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
GRAY = "\033[90m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

async def inspect():
    print(f"Connecting to {DATABASE_URL}...")
    try:
        conn = await asyncpg.connect(DATABASE_URL)
    except Exception as e:
        print(f"Error connecting: {e}")
        return

    try:
        print(f"{BOLD}Fetching clusters with > 1 items...{RESET}\n")
        
        # FIXED SQL: Includes item_count and individual item distances
        rows = await conn.fetch("""
            SELECT 
                c.id, 
                c.title_summary, 
                c.last_updated_at,
                COUNT(i.id) as item_count,
                json_agg(json_build_object(
                    'title', i.title, 
                    'content', i.content,
                    'source', s.url,
                    'distance', (i.embedding <=> c.centroid)
                ) ORDER BY (i.embedding <=> c.centroid) ASC) as items_data
            FROM clusters c
            JOIN items i ON i.cluster_id = c.id
            JOIN sources s ON i.source_id = s.id
            GROUP BY c.id
            HAVING COUNT(i.id) > 1
            ORDER BY c.last_updated_at DESC
            LIMIT 30;
        """)

        if not rows:
            print(f"{YELLOW}No clusters found with multiple items.{RESET}")
            return

        for row in rows:
            cluster_id = str(row['id'])
            summary = row['title_summary']
            count = row['item_count']
            last_upd = row['last_updated_at'].strftime('%Y-%m-%d %H:%M')

            print(f"{BOLD}{CYAN}CLUSTER: {summary}{RESET}")
            print(f"ID: {cluster_id} | Updated: {last_upd} | {BOLD}Size: {count}{RESET}")
            print("=" * 80)

            items = json.loads(row['items_data'])

            for item in items:
                title = item.get('title') or ""
                dist = item.get('distance') or 0.0
                source = item.get('source') or "Unknown"
                
                # Similarity is 1 - Distance
                similarity = 1 - dist
                
                # Color code the distance to help you tune
                # Distances > 0.10 are risky for "Same Event" news
                dist_color = GREEN if dist < 0.08 else (YELLOW if dist < 0.12 else RED)

                print(f"  • [{GREEN}{source}{RESET}] {BOLD}{title}{RESET}")
                print(f"    {dist_color}Distance: {dist:.4f} (Similarity: {similarity:.4f}){RESET}")
                
                if SHOW_INPUT_TEXT:
                    content = item.get('content') or ""
                    clean_content = content[:200].replace('\n', ' ').strip()
                    print(f"    {GRAY}Input: {title}. {clean_content}...{RESET}")
                
                print(f"    {GRAY}--------------------------------------------------{RESET}")
            
            print("\n")

    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(inspect())
