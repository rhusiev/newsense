CREATE TABLE IF NOT EXISTS item_predictions (
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    item_id UUID REFERENCES items(id) ON DELETE CASCADE,
    score REAL NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, item_id)
);

CREATE INDEX IF NOT EXISTS idx_item_predictions_user_id ON item_predictions(user_id);
CREATE INDEX IF NOT EXISTS idx_item_predictions_item_id ON item_predictions(item_id);
CREATE INDEX IF NOT EXISTS idx_item_predictions_score ON item_predictions(score);
