CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(255) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS remember_tokens (
    series UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)

CREATE TABLE IF NOT EXISTS feeds (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    owner_id UUID REFERENCES users(id) ON DELETE CASCADE,
    url TEXT UNIQUE NOT NULL,
    title VARCHAR(255),
    description TEXT,
    last_fetched_at TIMESTAMP WITH TIME ZONE,
    error_count INT DEFAULT 0,
    is_public BOOLEAN,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    feed_id UUID REFERENCES feeds(id) ON DELETE CASCADE,

    title TEXT NOT NULL,
    link TEXT NOT NULL,
    content TEXT,
    author VARCHAR(255),

    published_at TIMESTAMP WITH TIME ZONE,
    cluster_id UUID,
    embedding vector(384),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT unique_item_per_feed UNIQUE (feed_id, link)
);

CREATE TABLE IF NOT EXISTS item_reads (
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    item_id UUID REFERENCES items(id) ON DELETE CASCADE,
    is_read BOOLEAN NOT NULL DEFAULT true,
    liked REAL DEFAULT 0 CHECK (liked >= -1 AND liked <= 1),
    marked_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, item_id)
);

CREATE TABLE IF NOT EXISTS feed_subscriptions (
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    feed_id UUID REFERENCES feeds(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
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
    pattern TEXT NOT NULL,
);

CREATE TABLE IF NOT EXISTS rule_subscriptions (
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    rule_id UUID REFERENCES rules(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, rule_id)
);

-- For performance
CREATE INDEX idx_item_reads_user_id ON item_reads(user_id);
CREATE INDEX idx_item_reads_item_id ON item_reads(item_id);
CREATE INDEX idx_items_published ON items(published_at DESC);
CREATE INDEX idx_items_feed_id ON items(feed_id);
CREATE INDEX idx_feed_subscriptions_user_id ON feed_subscriptions(user_id);
CREATE INDEX idx_rule_subscriptions_user_id ON rule_subscriptions(user_id);
CREATE INDEX idx_items_cluster_id ON items(cluster_id) WHERE cluster_id IS NOT NULL;
-- Index for vector search (IVFFlat is good for speed/recall balance)
-- Note: You usually create this after you have some data, but here is the syntax:
-- CREATE INDEX ON items USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
