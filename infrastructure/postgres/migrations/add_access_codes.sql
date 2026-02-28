CREATE TABLE IF NOT EXISTS access_codes (
    name VARCHAR(255) PRIMARY KEY,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
