-- Per-feed switch to fetch each new article's page and keep its readable part.
-- Off by default. The fetcher skips sources that any public feed uses.
ALTER TABLE feeds ADD COLUMN IF NOT EXISTS full_text BOOLEAN NOT NULL DEFAULT false;

CREATE TABLE IF NOT EXISTS item_full_texts (
    item_id UUID PRIMARY KEY REFERENCES items(id) ON DELETE CASCADE,
    html TEXT,
    text TEXT,
    error TEXT,
    fetched_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
