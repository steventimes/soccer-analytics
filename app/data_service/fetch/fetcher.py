import os
import requests
import time
from dotenv import load_dotenv
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
import logging

from app.data_service.data_type import type_db_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API key and website
load_dotenv()
API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
BASE_URL = "https://api.football-data.org/v4/"


class RateLimiter:
    """
    Handle API rate limiting (10 requests per minute for my free plan)
    """
    def __init__(self, max_requests: int = 10, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.request_times: List[float] = []
    
    def wait_if_needed(self):
        now = time.time()
        
        self.request_times = [t for t in self.request_times if now - t < self.time_window]
        
        if len(self.request_times) >= self.max_requests:
            sleep_time = self.time_window - (now - self.request_times[0]) + 1
            logger.info(f"Rate limit reached. Waiting {sleep_time:.1f} seconds...")
            time.sleep(sleep_time)
            self.request_times = []
        
        self.request_times.append(time.time())

rate_limiter = RateLimiter()

def api_get(type, data) -> pd.DataFrame:
    """
    Enhanced API get function with match support
    Extends your existing function
    """
    match type:
        case type_db_data.TEAM_PLAYER:
            return get_team_players(data)
        case type_db_data.COMPETITION_STANDING:
            return get_competition_standings(data)
        case type_db_data.SINGLE_TEAM:
            return get_single_team(data)
        case type_db_data.COMPETITION_MATCHES:
            # data in tuple: (competition_code, season)
            if isinstance(data, tuple):
                if data[2]:
                    matches = get_competition_matches(data[0], data[1], data[2])
                else:
                    matches = get_competition_matches(data[0], data[1])
                return pd.DataFrame(matches)
            return pd.DataFrame()
        case type_db_data.TOP_SCORERS:
            # data in tuple: (competition_code, season)
            if isinstance(data, tuple):
                if data[2]:
                    scorers = get_competition_top_scorers(data[0], data[1], data[2])
                else:
                    scorers = get_competition_top_scorers(data[0], data[1])
                return pd.DataFrame(scorers)
            return pd.DataFrame()
    
    return pd.DataFrame()

def fetch(url: str, use_rate_limit: bool = True) -> Optional[Dict]:
    """
    Enhanced fetch with rate limiting
    
    Args:
        url: API endpoint URL
        use_rate_limit: Whether to apply rate limiting
        
    Returns:
        JSON response data or None if error
    """
    if use_rate_limit:
        rate_limiter.wait_if_needed()
    
    headers = {"X-Auth-Token": API_KEY}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if response.status_code == 429:
            logger.warning("Rate limit exceeded. Waiting 60 seconds...")
            time.sleep(60)
            return fetch(url, use_rate_limit=False)
        logger.error(f"HTTP Error fetching {url}: {e}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request Error: {e}")
        return None


def get_team_players(team_id: int) -> pd.DataFrame:
    """Fetch players from a specific team"""
    url = f"{BASE_URL}teams/{team_id}"
    data = fetch(url)
    if not data:
        return pd.DataFrame()
    players = data.get("squad", [])
    return pd.DataFrame(players)

def get_single_team(team_id: int) -> pd.DataFrame:
    """
    Fetch detailed information about a single team
    
    Args:
        team_id: Team ID
        
    Returns:
        DataFrame with team information (single row)
    """
    url = f"{BASE_URL}teams/{team_id}"
    data = fetch(url)
    if not data:
        return pd.DataFrame()
    
    # Extract main team info (exclude squad, staff, etc.)
    team_info = {
        'id': data.get('id'),
        'name': data.get('name'),
        'shortName': data.get('shortName'),
        'tla': data.get('tla'),
        'founded': data.get('founded'),
        'crest': data.get('crest'),
        'venue': data.get('venue'),
        'address': data.get('address'),
        'website': data.get('website'),
        'clubColors': data.get('clubColors'),
        'area_name': data.get('area', {}).get('name'),
        'area_id': data.get('area', {}).get('id'),
        'market_value': data.get('marketValue'),
    }
    
    # Add coach info if available
    coach = data.get('coach', {})
    if coach:
        team_info['coach_id'] = coach.get('id')
        team_info['coach_name'] = coach.get('name')
        team_info['coach_nationality'] = coach.get('nationality')
    
    return pd.DataFrame([team_info])

def get_competition_standings(competition_code="PL") -> pd.DataFrame:
    """Fetch competition standings"""
    url = f"{BASE_URL}competitions/{competition_code}/standings"
    data = fetch(url)
    if not data:
        return pd.DataFrame()
    standings = data["standings"][0]["table"]
    return pd.DataFrame(standings)

def get_competition_matches(competition_code: str, season: str, status: Optional[str] = None) -> List[Dict]:
    """
    Fetch all matches for a competition season
    
    Args:
        competition_code: e.g., 'PL', 'PD', 'BL1'
        season: e.g., '2023' for 2023/24 season
        status: Optional filter - 'FINISHED', 'SCHEDULED', etc.
        
    Returns:
        List of match dictionaries
    """
    url = f"{BASE_URL}competitions/{competition_code}/matches?season={season}"
    
    if status:
        url += f"&status={status}"
    
    logger.info(f"Fetching matches: {competition_code} {season}")
    
    data = fetch(url)
    if not data or 'matches' not in data:
        logger.error(f"No matches data for {competition_code} {season}")
        return []
    
    matches = data['matches']
    logger.info(f"Fetched {len(matches)} matches")
    
    return matches


def get_match_details(match_id: int) -> Optional[Dict]:
    """
    Fetch detailed information for a specific match
    Includes lineups, goals, bookings, etc.
    
    Args:
        match_id: Match ID
        
    Returns:
        Match details dictionary
    """
    url = f"{BASE_URL}matches/{match_id}"
    logger.info(f"Fetching match details: {match_id}")
    
    data = fetch(url)
    if not data:
        logger.error(f"Failed to fetch match {match_id}")
        return None
    
    return data


def get_competition_top_scorers(competition_code: str, season: str, limit: int = 10) -> List[Dict]:
    """
    Fetch top scorers for a competition season
    
    Args:
        competition_code: e.g., 'PL'
        season: e.g., '2023'
        limit: Number of top scorers to fetch
        
    Returns:
        List of scorer dictionaries
    """
    url = f"{BASE_URL}competitions/{competition_code}/scorers?season={season}&limit={limit}"
    
    logger.info(f"Fetching top scorers: {competition_code} {season}")
    
    data = fetch(url)
    if not data or 'scorers' not in data:
        logger.error(f"No scorers data for {competition_code} {season}")
        return []
    
    scorers = data['scorers']
    logger.info(f"Fetched {len(scorers)} scorers")
    
    return scorers


def get_team_matches(team_id: int, season: Optional[str] = None, status: str = 'FINISHED') -> List[Dict]:
    """
    Fetch matches for a specific team
    
    Args:
        team_id: Team ID
        season: Optional season year
        status: Match status filter
        
    Returns:
        List of match dictionaries
    """
    url = f"{BASE_URL}teams/{team_id}/matches?status={status}"
    
    if season:
        url += f"&season={season}"
    
    logger.info(f"Fetching matches for team {team_id}")
    
    data = fetch(url)
    if not data or 'matches' not in data:
        logger.error(f"No matches data for team {team_id}")
        return []
    
    return data['matches']


def fetch_multiple_seasons(
    competition_code: str,
    seasons: List[str],
    status: str = 'FINISHED'
) -> Dict[str, List[Dict]]:
    """
    Fetch multiple seasons with rate limiting
    
    Args:
        competition_code: Competition code
        seasons: List of season years ['2021', '2022', '2023']
        status: Match status to fetch
        
    Returns:
        Dictionary mapping season to matches
    """
    all_matches = {}
    total_matches = 0
    
    logger.info(f"Starting bulk fetch: {competition_code}, {len(seasons)} seasons")
    start_time = time.time()
    
    for i, season in enumerate(seasons, 1):
        logger.info(f"Progress: {i}/{len(seasons)} - Season {season}")
        
        matches = get_competition_matches(competition_code, season, status)
        all_matches[season] = matches
        total_matches += len(matches)
        
        logger.info(f"Season {season}: {len(matches)} matches")
    
    elapsed = time.time() - start_time
    logger.info(f"Bulk fetch complete!")
    logger.info(f"Total: {total_matches} matches in {elapsed:.1f} seconds")
    
    return all_matches


def fetch_initial_ml_dataset(
    competition_code: str = 'PL',
    num_seasons: int = 2
) -> List[Dict]:
    """
    Fetch initial dataset for ML training
    Fetches recent finished matches
    
    Args:
        competition_code: Competition to fetch
        num_seasons: Number of recent seasons
        
    Returns:
        List of all matches
    """
    current_year = datetime.now().year
    seasons = [str(current_year - i) for i in range(num_seasons)]
    
    logger.info(f"Fetching initial ML dataset")
    logger.info(f"Seasons: {seasons}")
    
    season_matches = fetch_multiple_seasons(competition_code, seasons, 'FINISHED')
    
    all_matches = []
    for season, matches in season_matches.items():
        all_matches.extend(matches)
    
    finished_matches = [
        m for m in all_matches 
        if m['status'] == 'FINISHED' and 
        m['score']['fullTime']['home'] is not None
    ]
    
    logger.info(f"Final dataset: {len(finished_matches)} finished matches")
    
    return finished_matches


def estimate_fetch_time(num_seasons: int, matches_per_season: int = 380) -> str:
    """
    Estimate how long fetching will take
    Note: API returns full season in 1 request, so it's actually fast!
    """
    requests_needed = num_seasons  # 1 request per season
    minutes = requests_needed / 10  # 10 requests per minute
    
    if minutes < 1:
        return f"~{minutes * 60:.0f} seconds"
    elif minutes < 60:
        return f"~{minutes:.1f} minutes"
    else:
        hours = minutes / 60
        return f"~{hours:.1f} hours"


def print_fetch_plan(competition_code: str, num_seasons: int):
    """Print a fetch plan"""
    print("=" * 60)
    print(f"Fetch Plan: {competition_code}")
    print("=" * 60)
    print(f"Seasons to fetch: {num_seasons}")
    print(f"Estimated matches: ~{num_seasons * 380}")
    print(f"Estimated time: {estimate_fetch_time(num_seasons)}")
    print(f"Rate limit: 10 requests/minute (handled automatically)")
    print("=" * 60)
