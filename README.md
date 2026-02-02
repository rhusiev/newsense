```sh
docker network create web_gateway
```

```sh
cargo install sqlx-cli
cargo sqlx prepare
```

```sh
docker exec -it newsense_db psql -U newsense_user -d newsense_db
```

```sql
BEGIN;
UPDATE items 
SET embedding = NULL, 
    cluster_id = NULL;
DELETE FROM clusters;
COMMIT;
```
