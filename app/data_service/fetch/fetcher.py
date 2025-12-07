import os
import requests
import time
import logging
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv

from app.data_service.db.cache.cache_management import get_redis_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
BASE_URL = "https://api.football-data.org/v4/"
redis_client = get_redis_client()

class RedisRateLimiter:
    """
    Persist rate limits in Redis using a Sliding Window algorithm.
    Allows 10 requests per minute (Free Tier).
    """
    def __init__(self, key_prefix: str = "rate_limit:football_api", limit: int = 10, window: int = 60):
        self.key = key_prefix
        self.limit = limit
        self.window = window

    def wait_if_needed(self):
        """
        Checks the number of requests in the last 'window' seconds.
        If limit is reached, sleeps until the oldest request expires.
        """
        now = time.time()
        pipeline = redis_client.pipeline()
        
        # reset time window
        pipeline.zremrangebyscore(self.key, 0, now - self.window)
        pipeline.zcard(self.key)
        pipeline.zrange(self.key, 0, 0)
        _, count, oldest_timestamp = pipeline.execute()

        if count >= self.limit:
            sleep_time = self.window - (now - float(oldest_timestamp[0])) + 0.5
            if sleep_time > 0:
                logger.warning(f"Rate limit hit ({self.limit}/{self.limit}). Sleeping {sleep_time:.2f}s...")
                time.sleep(sleep_time)

    def add_request(self):
        now = time.time()
        pipeline = redis_client.pipeline()
        pipeline.zadd(self.key, {str(now): now})
        pipeline.expire(self.key, self.window)
        pipeline.execute()

limiter = RedisRateLimiter()

def api_get(endpoint: str, params: Dict = None) -> Optional[Dict]:
    limiter.wait_if_needed()
    
    headers = {'X-Auth-Token': API_KEY}
    url = f"{BASE_URL}{endpoint}"
    
    try:
        response = requests.get(url, headers=headers, params=params)
        limiter.add_request()
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            logger.warning(f"API Rate Limit 429. Sleeping 60s...")
            time.sleep(60)
            return api_get(endpoint, params) # Retry
        elif response.status_code in [403, 404]:
            logger.warning(f"{response.status_code} Forbidden: Plan restriction or invalid key for {url}")
            return None
        else:
            logger.error(f"API Error {response.status_code}: {url}")
            return None
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return None

def fetch_multiple_seasons(competition_code: str, seasons: List[str]) -> Dict:
    """
    Fetch matches for multiple seasons. 
    Returns a dict keyed by season year.
    """
    all_seasons_data = {}
    
    for season in seasons:
        logger.info(f"Fetching {competition_code} season {season}...")

        data = api_get(f"competitions/{competition_code}/matches", {'season': season})

        if data and 'matches' in data:
            matches = data['matches']
            
            valid_matches = []
            for m in matches:
                if m['status'] == 'FINISHED' and m.get('score', {}).get('fullTime', {}).get('home') is not None:
                    valid_matches.append(m)
            
            all_seasons_data[season] = valid_matches
            logger.info(f"  -> Retrieved {len(valid_matches)} valid matches.")
        else:
            logger.warning(f"  -> No data found for {season} (likely older than plan allows).")
            
    return all_seasons_data

def fetch_competition_details(competition_code: str) -> Optional[Dict]:
    """
    Fetch metadata for a specific competition (e.g., PL, CL).
    """
    logger.info(f"Fetching details for competition: {competition_code}")
    return api_get(f"competitions/{competition_code}")

def fetch_team_squad(team_id: int) -> Optional[Dict]:
    """
    Fetch full squad details for a specific team.
    """
    logger.info(f"Fetching squad for Team ID: {team_id}")
    return api_get(f"teams/{team_id}")

def fetch_standings(competition_code: str, season: str) -> Optional[Dict]:
    """
    Get the league table for a specific season.
    """
    logger.info(f"Fetching Standings: {competition_code} {season}")
    return api_get(f"competitions/{competition_code}/standings", {'season': season})

def fetch_top_scorers(competition_code: str, season: str) -> Optional[Dict]:
    """
    Get the top scorers list for a specific season.
    """
    logger.info(f"Fetching Top Scorers: {competition_code} {season}")
    return api_get(f"competitions/{competition_code}/scorers", {'season': season})