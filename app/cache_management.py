import json
import redis
from datetime import timedelta
from typing import Optional, Any

# Connect to Redis
redis_client = redis.Redis(
    host="redis",  # matches service name in docker-compose.yml
    port=6379,
    db=0,
    decode_responses=True  # ensures we get strings instead of bytes
)

TTL = timedelta(minutes=5).seconds  # 5-minute cache

def get_cache(key: str) -> Optional[Any]:
    """Retrieve JSON value from cache."""
    try: 
        value = redis_client.get(key)
        if value:
            return json.loads(value)
    except (redis.RedisError, json.JSONDecodeError) as e:
        print(f"Cache fetch error, key: '{key}'. Error:'{e}'")
    return None

def set_cache(key: str, value, ttl: int = TTL):
    """Store Python object as JSON in cache."""
    redis_client.setex(key, ttl, json.dumps(value))

def get_or_fetch_team_players(team_id: int, fetch_func):
    """
    Check cache first. If not found, call fetch_func(team_id),
    store it, and return the result.
    """
    key = f"team:{team_id}:players"
    cached = get_cache(key)
    if cached:
        print(f"Cache hit for {key}")
        return cached

    print(f"Cache miss for {key}, fetching from DB/API...")
    data = fetch_func(team_id)
    if data is not None:
        set_cache(key, data)
    return data
