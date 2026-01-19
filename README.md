```sh
docker network create web_gateway
```

```sql
BEGIN;
UPDATE items 
SET embedding = NULL, 
    cluster_id = NULL;
DELETE FROM clusters;
COMMIT;
```
