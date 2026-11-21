-- A read item has no label until the user rates it. Bulk mark-read used to store
-- 0.0 (neutral) for every item it touched, and training counts every non-null label.
ALTER TABLE item_reads ALTER COLUMN liked DROP DEFAULT;

-- Undo those bulk labels. A bulk mark-read stamps all its rows with one marked_at;
-- more than 20 rows sharing it is taken as bulk. 20 is chosen, not derived: a
-- cluster rated at once also shares a stamp, and the largest cluster seen on the
-- main instance was 11 items (2026-11-19).
UPDATE item_reads ir
SET liked = NULL
FROM (
    SELECT user_id, marked_at
    FROM item_reads
    GROUP BY user_id, marked_at
    HAVING COUNT(*) > 20
) bulk
WHERE ir.user_id = bulk.user_id
  AND ir.marked_at = bulk.marked_at
  AND ir.liked = 0;
