ALTER TABLE items ADD COLUMN IF NOT EXISTS preference_embedding vector(768);
CREATE INDEX IF NOT EXISTS idx_items_pref_embedding_hnsw ON items USING hnsw (preference_embedding vector_cosine_ops);
