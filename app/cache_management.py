import redis
import json

def init_cache(host="localhost", port=6379, db=0):
    return redis.StrictRedis(host=host, port=port, db=db, decode_responses=True)

def cache_players(r, team_id, df):
    key = f"team:{team_id}:players"
    r.set(key, df.to_json())

def get_cached_players(r, team_id):
    key = f"team:{team_id}:players"
    data = r.get(key)
    if data:
        import pandas as pd
        return pd.read_json(data)
    return None
