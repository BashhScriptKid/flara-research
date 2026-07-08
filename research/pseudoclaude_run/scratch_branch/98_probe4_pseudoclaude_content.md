The problem isn’t the scaling itself—it’s that your global in‑memory cache now spans multiple instances, and each instance is holding onto **live ORM objects** that become stale when the database session closes.

Here’s what’s happening:

- `fetch_profile_from_db` runs `db.query(...)` and returns an ORM model instance (likely from SQLAlchemy, or something similar).  
- That instance is still attached to the database session that opened during the request.  
- Once the request finishes, Flask (or your WSGI server) tears down the session.  
- On the *next* request for the same user hitting the same instance, `jsonify(request_cache[user_id])` tries to serialize the now‑detached ORM object.  
- Serialization often causes lazy‑loaded attributes (relationships, columns not pre‑loaded) to be fetched—triggering a new query against the already‑closed session. That throws a `DetachedInstanceError` or similar, which Flask turns into a 500.

With a single instance, the session scope might have accidentally lined up (e.g., a long‑lived session, or no teardown), so you never saw the issue. With three instances, each one builds its own cache independently, and any request that lands on an instance that cached the object earlier will hit the stale‑object error *randomly*—exactly the 1‑in‑200 pattern you’re seeing.

**The fix:** convert the database result to a plain dictionary (or a JSON‑serializable structure) *before* you put it in the cache. That way the cached value is fully independent of the DB session.

```python
request_cache = {}

@app.route("/api/profile/<user_id>")
def get_profile(user_id):
    if user_id not in request_cache:
        raw = fetch_profile_from_db(user_id)
        # Convert to a plain dict so it can be safely serialised later
        request_cache[user_id] = dict(raw)   # or raw._asdict(), etc.
    return jsonify(request_cache[user_id])
```

If your ORM object is a dict‑like row, `dict(raw)` works. If it’s a custom object, define a `.to_dict()` method or use `{col: getattr(raw, col) for col in columns}`. Choose whatever gives you a plain, detached data structure.

**Also important (but not the cause of the 500s):** your query is vulnerable to SQL injection because it uses f‑strings directly in the SQL. Always parameterise:  
```python
db.query("SELECT * FROM profiles WHERE id = %s", (user_id,))
```
or, better, use your ORM’s safe query builder (e.g., `Profile.query.get(user_id)`).

**Longer term:** to truly fix the scaling issue, replace the per‑instance dict with a shared cache (Redis, Memcached). That way every instance sees the same cached data, you avoid redundant DB hits, and you never serve a stale ORM object. But the immediate stopgap above will eliminate the 500 errors.