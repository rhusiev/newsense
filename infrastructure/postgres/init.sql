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
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS clusters (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    centroid vector(384),
    title_summary TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID REFERENCES sources(id) ON DELETE CASCADE,

    title TEXT NOT NULL,
    link TEXT NOT NULL,
    content TEXT,
    author VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    published_at TIMESTAMP WITH TIME ZONE,
    cluster_id UUID,
    embedding vector(384),
    CONSTRAINT fk_items_clusters FOREIGN KEY (cluster_id) REFERENCES clusters(id),

    CONSTRAINT unique_item_per_source UNIQUE (source_id, link)
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
    pattern TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rule_subscriptions (
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    rule_id UUID REFERENCES rules(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, rule_id)
);

CREATE INDEX idx_item_reads_user_id ON item_reads(user_id);
CREATE INDEX idx_item_reads_item_id ON item_reads(item_id);
CREATE INDEX idx_items_published ON items(published_at DESC);
CREATE INDEX idx_items_source_id ON items(source_id);
CREATE INDEX idx_feeds_source_id ON feeds(source_id);
CREATE INDEX idx_feed_subscriptions_user_id ON feed_subscriptions(user_id);
CREATE INDEX idx_items_cluster_id ON items(cluster_id) WHERE cluster_id IS NOT NULL;
CREATE INDEX idx_clusters_last_updated ON clusters(last_updated_at);

CREATE INDEX idx_items_embedding_hnsw ON items USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_clusters_centroid_hnsw ON clusters USING hnsw (centroid vector_cosine_ops);
