import json
import redis
from datetime import timedelta
from typing import Optional, Any, cast
import asyncio

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
        # redis-py's type stubs use a generic ResponseT which can confuse static
        # type checkers. Cast to str/bytes explicitly so json.loads accepts it.
        if value is not None:
            return json.loads(cast(str, value))
    except (redis.RedisError, json.JSONDecodeError) as e:
        print(f"Cache fetch error, key: '{key}'. Error:'{e}'")
    return None

def set_cache(key: str, value, ttl: int = TTL) -> bool:
    """Store Python object as JSON in cache."""
    try:
        redis_client.setex(key, ttl, json.dumps(value))
        return True
    except (redis.RedisError, json.JSONDecodeError) as e:
        print(f"Cache set error, key: '{key}'. Error:'{e}'")
    return False

def delete_cache(key: str) -> bool:
    '''delete a specific cache key'''
    try:
        # redis_client.delete may return a ResponseT; cast to int first for
        # predictable boolean conversion for the type checker.
        result = redis_client.delete(key)
        return bool(cast(int, result))
    except redis.RedisError as e:
        print(f"Cache delete error, key: '{key}'. Error:'{e}'")
        return False

def clear_all_pattern(pattern: str) -> int:
    '''delete all the keys matching the pattern'''
    try:
        keys = redis_client.keys(pattern)
        # keys may be typed as a generic ResponseT; cast to list for iteration
        keys_list = cast(list, keys) or []
        if keys_list:
            result = redis_client.delete(*keys_list)
            return int(cast(int, result)) if result else 0
        return 0
    except redis.RedisError as e:
        print(f"Cache delete pattern error, pattern: '{pattern}'. Error:'{e}'")
        return 0

def cache_exist(key: str) -> bool:
    try:
        result = redis_client.exists(key)
        # cast to int for the type checker
        return int(cast(int, result)) > 0
    except redis.RedisError as e:
        print(f"Cache exists check error for key '{key}': {e}")
        return False
        
def get_or_fetch_team_players(team_id: int, fetch_func):
    """
    Deprecated: use dataservice methods
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
        set_cache(key, data, TTL)
    return data
