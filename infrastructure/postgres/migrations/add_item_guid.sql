-- Items are identified by their feed-level guid rather than their link: feeds such
-- as Acast reuse one show-homepage link across every episode, which made the old
-- (source_id, link) constraint collapse the whole feed into a single item.
ALTER TABLE items ADD COLUMN IF NOT EXISTS guid TEXT;

ALTER TABLE items DROP CONSTRAINT IF EXISTS unique_item_per_source;
ALTER TABLE items ADD CONSTRAINT unique_item_per_source UNIQUE (source_id, guid);

CREATE INDEX IF NOT EXISTS idx_items_source_link ON items(source_id, link);
