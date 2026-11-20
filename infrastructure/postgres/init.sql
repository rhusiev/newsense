CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(255) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS access_codes (
    code VARCHAR(255) PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS remember_tokens (
    series UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    url TEXT UNIQUE NOT NULL,
    last_fetched_at TIMESTAMP WITH TIME ZONE,
    error_count INT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS feeds (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    owner_id UUID REFERENCES users(id) ON DELETE CASCADE,
    source_id UUID NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    title VARCHAR(255),
    description TEXT,
    is_public BOOLEAN DEFAULT false,
    full_text BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Story threads: clusters that follow one event over several days.
CREATE TABLE IF NOT EXISTS threads (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_story_at TIMESTAMP WITH TIME ZONE NOT NULL
);

CREATE TABLE IF NOT EXISTS clusters (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    centroid vector(384),
    title_summary TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    -- NULL thread_id with threads_checked_at set: a story with no thread yet.
    thread_id UUID REFERENCES threads(id) ON DELETE SET NULL,
    threads_checked_at TIMESTAMP WITH TIME ZONE
);

CREATE TABLE IF NOT EXISTS items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID REFERENCES sources(id) ON DELETE CASCADE,

    guid TEXT,
    title TEXT NOT NULL,
    link TEXT NOT NULL,
    content TEXT,
    author VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    published_at TIMESTAMP WITH TIME ZONE,
    cluster_id UUID,
    embedding vector(384),
    CONSTRAINT fk_items_clusters FOREIGN KEY (cluster_id) REFERENCES clusters(id),

    CONSTRAINT unique_item_per_source UNIQUE (source_id, guid)
);

CREATE TABLE IF NOT EXISTS item_reads (
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    item_id UUID REFERENCES items(id) ON DELETE CASCADE,
    is_read BOOLEAN NOT NULL DEFAULT true,
    liked REAL CHECK (liked >= -1 AND liked <= 1),
    marked_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, item_id)
);

CREATE TABLE IF NOT EXISTS user_preference_models (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    model_state BYTEA NOT NULL,
    last_trained_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    training_cursor TIMESTAMP WITH TIME ZONE
);

CREATE TABLE IF NOT EXISTS feed_subscriptions (
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    feed_id UUID REFERENCES feeds(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    -- The feed's own filter: 'off' shows every article; 'score' hides below
    -- score_threshold; 'top' keeps top_n a day. Used when the feed is opened and by
    -- All Articles in 'each' mode.
    filter_mode TEXT NOT NULL DEFAULT 'off' CHECK (filter_mode IN ('off', 'score', 'top')),
    score_threshold REAL CHECK (score_threshold >= -1 AND score_threshold <= 1),
    top_n INTEGER CHECK (top_n BETWEEN 1 AND 200),
    CONSTRAINT feed_filter_complete CHECK (
        (filter_mode <> 'score' OR score_threshold IS NOT NULL) AND (filter_mode <> 'top' OR top_n IS NOT NULL)),
    PRIMARY KEY (user_id, feed_id)
);

CREATE TABLE IF NOT EXISTS rules (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    owner_id UUID REFERENCES users(id) ON DELETE CASCADE,
    rule_type VARCHAR(50) NOT NULL,
    is_public BOOLEAN,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS regex_rules (
    id UUID REFERENCES rules(id) ON DELETE CASCADE,
    pattern TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rule_subscriptions (
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    rule_id UUID REFERENCES rules(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, rule_id)
);

CREATE TABLE IF NOT EXISTS item_predictions (
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    item_id UUID REFERENCES items(id) ON DELETE CASCADE,
    score REAL NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, item_id)
);

CREATE TABLE IF NOT EXISTS exposures (
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    entity_id UUID NOT NULL,
    day DATE NOT NULL,
    policy TEXT NOT NULL CHECK (policy IN ('top', 'boundary', 'random')),
    propensity REAL NOT NULL CHECK (propensity > 0 AND propensity <= 1),
    rank INTEGER NOT NULL,
    score REAL NOT NULL,
    shown_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, entity_id)
);

CREATE TABLE IF NOT EXISTS item_full_texts (
    item_id UUID PRIMARY KEY REFERENCES items(id) ON DELETE CASCADE,
    html TEXT,
    text TEXT,
    error TEXT,
    fetched_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_settings (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    -- All Articles: 'off' shows every story; 'score' hides below score_threshold;
    -- 'top' keeps each day's top_per_day best stories; 'each' applies each feed's own filter
    all_articles_filter TEXT NOT NULL DEFAULT 'score' CHECK (all_articles_filter IN ('off', 'score', 'top', 'each')),
    score_threshold REAL NOT NULL DEFAULT -0.7 CHECK (score_threshold >= -1 AND score_threshold <= 1),
    top_per_day INTEGER NOT NULL DEFAULT 40 CHECK (top_per_day BETWEEN 1 AND 200),
    use_clusters BOOLEAN NOT NULL DEFAULT true,
    -- 'sheet': reader opens over the list; 'page': reader replaces the list
    reader_layout TEXT NOT NULL DEFAULT 'sheet' CHECK (reader_layout IN ('sheet', 'page')),
    -- Opt-in LLM summaries of a story; the switch is the user's, the limit the admin's.
    llm_summaries BOOLEAN NOT NULL DEFAULT false,
    llm_allowed BOOLEAN NOT NULL DEFAULT true,
    llm_daily_limit INTEGER NOT NULL DEFAULT 70 CHECK (llm_daily_limit BETWEEN 1 AND 10000),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_item_predictions_user_id ON item_predictions(user_id);
CREATE INDEX idx_item_predictions_item_id ON item_predictions(item_id);
CREATE INDEX idx_item_predictions_score ON item_predictions(score);

CREATE INDEX idx_item_reads_user_id ON item_reads(user_id);
CREATE INDEX idx_item_reads_item_id ON item_reads(item_id);
CREATE INDEX idx_items_published ON items(published_at DESC);
CREATE INDEX idx_items_source_id ON items(source_id);
CREATE INDEX idx_items_source_link ON items(source_id, link);
CREATE INDEX idx_feeds_source_id ON feeds(source_id);
CREATE INDEX idx_feed_subscriptions_user_id ON feed_subscriptions(user_id);
CREATE INDEX idx_feed_subscriptions_feed_id ON feed_subscriptions(feed_id);
CREATE INDEX idx_items_cluster_id ON items(cluster_id) WHERE cluster_id IS NOT NULL;
CREATE INDEX idx_clusters_thread_id ON clusters(thread_id) WHERE thread_id IS NOT NULL;
CREATE INDEX idx_clusters_threads_unchecked ON clusters(created_at) WHERE threads_checked_at IS NULL;
CREATE INDEX idx_clusters_last_updated ON clusters(last_updated_at);

CREATE INDEX idx_items_embedding_hnsw ON items USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_clusters_centroid_hnsw ON clusters USING hnsw (centroid vector_cosine_ops);

-- Stories that appeared in a user's list, for the automatic summary job.
CREATE TABLE IF NOT EXISTS story_impressions (
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    story_id UUID NOT NULL,
    first_shown_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, story_id)
);
CREATE INDEX IF NOT EXISTS idx_story_impressions_user_shown
    ON story_impressions(user_id, first_shown_at);


-- One cached summary per user and story, with failed attempts backed off.
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
CREATE INDEX IF NOT EXISTS idx_llm_calls_user_day ON llm_calls(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_story_summaries_pending
    ON story_summaries(updated_at) WHERE summary IS NULL;

-- A user's own model endpoint and key, encrypted at rest.
CREATE TABLE IF NOT EXISTS user_llm_keys (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    base_url TEXT NOT NULL,
    model TEXT NOT NULL,
    key_ciphertext BYTEA NOT NULL,
    key_nonce BYTEA NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
