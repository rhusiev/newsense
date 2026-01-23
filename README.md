```sh
docker network create web_gateway
```

```sh
cargo install sqlx-cli
cargo sqlx prepare
```

```sql
BEGIN;
UPDATE items 
SET embedding = NULL, 
    cluster_id = NULL;
DELETE FROM clusters;
COMMIT;
```
