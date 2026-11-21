-- A user's own OpenAI-compatible endpoint and key. The key is encrypted with
-- LLM_KEY_ENCRYPTION_KEY (32 bytes, base64) and is never returned by the API.
CREATE TABLE IF NOT EXISTS user_llm_keys (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    base_url TEXT NOT NULL,
    model TEXT NOT NULL,
    key_ciphertext BYTEA NOT NULL,
    key_nonce BYTEA NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
