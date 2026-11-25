-- Each thread-verifier request, successful or not, for the daily cap.
CREATE TABLE IF NOT EXISTS thread_llm_calls (
    id BIGSERIAL PRIMARY KEY,
    success BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_thread_llm_calls_created ON thread_llm_calls(created_at);
-- The daily cap no longer counts cached verdicts, so their index has no reader.
DROP INDEX IF EXISTS idx_thread_pair_scores_created;
