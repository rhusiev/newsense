-- Opt-in LLM summaries of a story. `llm_summaries` is the user's switch;
-- `llm_allowed` and the daily limit are admin controlled.
ALTER TABLE user_settings
    ADD COLUMN IF NOT EXISTS llm_summaries BOOLEAN NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS llm_allowed BOOLEAN NOT NULL DEFAULT true,
    ADD COLUMN IF NOT EXISTS llm_daily_limit INTEGER NOT NULL DEFAULT 70
        CHECK (llm_daily_limit BETWEEN 1 AND 10000);

-- One cached summary per user and story. A failed attempt keeps the row with
-- `error` and an attempt count, so the worker backs off instead of spending the
-- daily budget on the same story.
CREATE TABLE IF NOT EXISTS story_summaries (
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    story_id UUID NOT NULL,
    summary TEXT,
    error TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    source_hash TEXT,
    model TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, story_id)
);

-- Every model call, for the daily limit and for cost accounting.
CREATE TABLE IF NOT EXISTS llm_calls (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    story_id UUID NOT NULL,
    model TEXT NOT NULL,
    success BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_llm_calls_user_day
    ON llm_calls(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_story_summaries_pending
    ON story_summaries(updated_at) WHERE summary IS NULL;
