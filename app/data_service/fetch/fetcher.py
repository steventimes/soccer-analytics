import os
import requests
import time
import logging
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv

# Import the shared Redis client from your cache module
# This ensures we use the same connection defined in cache_management.py
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
        # Count requests in the current window
        pipeline.zcard(self.key)
        # Get the timestamp of the oldest in the window
        pipeline.zrange(self.key, 0, 0, withscores=True)
        
        _, current_count, oldest_request = pipeline.execute()

        if current_count >= self.limit:
            # Calculate sleep time
            if oldest_request:
                oldest_ts = oldest_request[0][1]
                sleep_time = self.window - (now - oldest_ts) + 1
            else:
                sleep_time = 1
            
            logger.warning(f"Rate limit hit ({current_count}/{self.limit}). Sleeping {sleep_time:.2f}s...")
            time.sleep(sleep_time)
            
            # Recursive check
            self.wait_if_needed()
        else:
            # Add current request timestamp
            redis_client.zadd(self.key, {str(now): now})
            # Set key expiry to clean up
            redis_client.expire(self.key, self.window + 10)

def api_get(endpoint: str, params: Dict = {}) -> Optional[Any]:
    """
    Centralized API requester with Error Handling and Redis Rate Limiting.
    """
    limiter = RedisRateLimiter()
    url = f"{BASE_URL}{endpoint}"
    headers = {'X-Auth-Token': API_KEY}
    
    # Retry for server errors
    max_retries = 3
    for attempt in range(max_retries):
        limiter.wait_if_needed()
        
        try:
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                return response.json()
          
            # Too Many Requests: Retry 
            elif response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                logger.warning(f"429 Received. Waiting {retry_after}s before retry.")
                time.sleep(retry_after)
                continue
                
            # Don't retry 403
            elif response.status_code == 403:
                logger.warning(f"403 Forbidden: Plan restriction or invalid key for {url}")
                return None
                
            else:
                logger.error(f"API Error {response.status_code}: {url}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error on {url}: {e}")
            time.sleep(2)
            
    return None

def fetch_multiple_seasons(competition_code: str, seasons: List[str], status: str = 'FINISHED') -> Dict[str, List[Dict]]:
    """
    Fetch matches for multiple seasons with safe parsing.
    """
    all_seasons_data = {}
    
    for season in seasons:
        logger.info(f"Fetching {competition_code} season {season}...")

        data = api_get(f"competitions/{competition_code}/matches", {'season': season, 'status': status})
        
        if data and 'matches' in data:
            matches = data['matches']
            valid_matches = []
            
            for m in matches:
                score = m.get('score', {}).get('fullTime', {})
                if score.get('home') is not None and score.get('away') is not None:
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