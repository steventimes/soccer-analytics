import os
import requests
import time
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv

from app.data_service.db.cache.cache_management import get_redis_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class RedisRateLimiter:
    """
    Persist rate limits in Redis using a Sliding Window algorithm.
    """
    def __init__(self, key_prefix: str = "rate_limit:football_api", limit: int = 10, window: int = 60):
        self.redis = get_redis_client()
        self.key = key_prefix
        self.limit = limit
        self.window = window

    def wait_if_needed(self):
        try:
            now = time.time()
            pipeline = self.redis.pipeline()
            pipeline.zremrangebyscore(self.key, 0, now - self.window)
            pipeline.zcard(self.key)
            pipeline.zrange(self.key, 0, 0)
            _, count, oldest = pipeline.execute()

            if count >= self.limit:
                oldest_ts = float(oldest[0]) if oldest else now
                sleep_time = self.window - (now - oldest_ts) + 0.5
                if sleep_time > 0:
                    logger.warning(f"Rate limit hit ({self.limit}/{self.limit}). Sleeping {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
        except Exception as e:
            logger.error(f"Rate Limiter Error: {e}. Proceeding without delay.")

    def add_request(self):
        try:
            now = time.time()
            pipeline = self.redis.pipeline()
            pipeline.zadd(self.key, {str(now): now})
            pipeline.expire(self.key, self.window)
            pipeline.execute()
        except Exception:
            pass

class FootballDataClient:
    def __init__(self):
        self.api_key = os.getenv("FOOTBALL_DATA_API_KEY")
        self.base_url = "https://api.football-data.org/v4/"
        self.limiter = RedisRateLimiter()
        
        if not self.api_key:
            logger.error("No API Key found! Set FOOTBALL_DATA_API_KEY in .env")

    def _get(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        self.limiter.wait_if_needed()
        
        headers = {'X-Auth-Token': self.api_key}
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = requests.get(url, headers=headers, params=params)
            self.limiter.add_request()
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                logger.warning(f"API Rate Limit 429. Sleeping 60s...")
                time.sleep(60)
                return self._get(endpoint, params)
            elif response.status_code in [403, 404]:
                logger.warning(f"{response.status_code} Error: {url}")
                return None
            else:
                logger.error(f"API Error {response.status_code}: {url}")
                return None
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None

    def fetch_multiple_seasons(self, competition_code: str, seasons: List[str]) -> Dict:
        all_seasons_data = {}
        for season in seasons:
            logger.info(f"Fetching {competition_code} season {season}...")
            data = self._get(f"competitions/{competition_code}/matches", {'season': season})
            
            if data and 'matches' in data:
                valid = [
                    m for m in data['matches'] 
                    if m['status'] == 'FINISHED' and m.get('score', {}).get('fullTime', {}).get('home') is not None
                ]
                all_seasons_data[season] = valid
                logger.info(f"  -> Retrieved {len(valid)} valid matches.")
            else:
                logger.warning(f"  -> No data found for {season}.")
        return all_seasons_data

    def fetch_competition_details(self, code: str):
        return self._get(f"competitions/{code}")

    def fetch_team_squad(self, team_id: int):
        return self._get(f"teams/{team_id}")

    def fetch_standings(self, code: str, season: str):
        return self._get(f"competitions/{code}/standings", {'season': season})

    def fetch_top_scorers(self, code: str, season: str):
        return self._get(f"competitions/{code}/scorers", {'season': season})