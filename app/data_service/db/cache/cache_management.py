import json
import redis
from redis.connection import ConnectionPool
import os
from dotenv import load_dotenv
from datetime import timedelta
from typing import Optional, Any, cast

load_dotenv()
# Connect to Redis
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", 6379))
redis_password = os.getenv("REDIS_PASSWORD")
redis_db = int(os.getenv("REDIS_DB", 0))
redis_pool = ConnectionPool(
    host=redis_host,
    port=redis_port,
    password=redis_password,
    db=redis_db,
    decode_responses=True,
    max_connections=10
)
# 5-minute cache
TTL = timedelta(minutes=5).seconds 

def get_redis_client():
    return redis.Redis(connection_pool=redis_pool)

def check_redis_health() -> bool:
    """Check if Redis is available"""
    try:
        client = get_redis_client()
        client.ping()
        return True
    except Exception as e:
        print(f"Redis health check failed: {e}")
        return False

def get_cache(key: str) -> Optional[Any]:
    """Retrieve JSON value from cache."""
    try: 
        redis_client = get_redis_client()
        value = redis_client.get(key)
        if value is not None:
            return json.loads(cast(str, value))
    except (redis.RedisError, json.JSONDecodeError) as e:
        print(f"Cache fetch error, key: '{key}'. Error:'{e}'")
    return None

def set_cache(key: str, value, ttl: int = TTL) -> bool:
    """Store Python object as JSON in cache."""
    if not check_redis_health():
        return False
    try:
        redis_client = get_redis_client()
        redis_client.setex(key, ttl, json.dumps(value))
        return True
    except (redis.RedisError, json.JSONDecodeError) as e:
        print(f"Cache set error, key: '{key}'. Error:'{e}'")
    return False

def delete_cache(key: str) -> bool:
    '''delete a specific cache key'''
    try:
        redis_client = get_redis_client()
        result = redis_client.delete(key)
        return bool(cast(int, result))
    except redis.RedisError as e:
        print(f"Cache delete error, key: '{key}'. Error:'{e}'")
        return False

def clear_all_pattern(pattern: str) -> int:
    '''delete all the keys matching the pattern'''
    try:
        redis_client = get_redis_client()
        keys = redis_client.keys(pattern)
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
        redis_client = get_redis_client()
        result = redis_client.exists(key)
        return int(cast(int, result)) > 0
    except redis.RedisError as e:
        print(f"Cache exists check error for key '{key}': {e}")
        return False

