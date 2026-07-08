The issue is almost certainly that `fetch_profile_from_db` returns an ORM object (like a SQLAlchemy model instance) which gets cached. On the first request for a given `user_id`, the object is alive because the database session still exists, and `jsonify` works. On subsequent requests to the **same** instance, the session from the original request is long gone, the object is detached, and trying to serialize it triggers lazy-loading of attributes – which fails with something like `DetachedInstanceError`, causing a 500. With a single instance you never saw it because the load balancer wasn’t spreading requests, so the same user rarely hit the cache after session closure; with 3 instances, the pattern becomes worse and manifests as intermittent errors.

**Fix:** Cache only plain dictionaries (direct JSON‑serializable data), never ORM objects. For example:

```python
request_cache = {}

@app.route("/api/profile/<user_id>")
def get_profile(user_id):
    if user_id not in request_cache:
        # Fetch the data and convert to a dict immediately
        row = fetch_profile_from_db(user_id)
        # Convert row to a plain dict – adapt based on what your db layer returns
        if row:
            request_cache[user_id] = dict(row)   # works with SQLAlchemy Row, or
            # request_cache[user_id] = {k: v for k, v in row.items()}
        else:
            request_cache[user_id] = {}           # handle missing user
    return jsonify(request_cache[user_id])

def fetch_profile_from_db(user_id):
    # ⚠️ Also fix the SQL injection vulnerability – use parameters
    result = db.execute("SELECT * FROM profiles WHERE id = %(id)s", {"id": user_id})
    return result.fetchone()   # returns a RowMapping/dict-like object
```

If you’re using SQLAlchemy ORM and need the full model elsewhere, fetch the object but then immediately extract the relevant attributes into a dict before caching. And please **never** build SQL queries with string interpolation – that’s a separate security bug waiting to explode.