import requests
import json
import logging
import re
from bs4 import BeautifulSoup
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class UnderstatClient:
    BASE_URL = "https://understat.com/league"

    def fetch_season_data(self, league_name: str, season_year: str) -> List[Dict]:
        url = f"{self.BASE_URL}/{league_name}/{season_year}"
        logger.info(f"Scraping Understat: {url}")
        
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            scripts = soup.find_all('script')
            
            for script in scripts:
                if 'datesData' in script.text:
                    match = re.search(r"JSON\.parse\('([^']+)'\)", script.text)
                    if match:
                        json_string = match.group(1)
                        json_string = json_string.encode('utf8').decode('unicode_escape')
                        return json.loads(json_string)
            
            return []
        except Exception as e:
            logger.error(f"Error scraping Understat: {e}")
            return []

    def fetch_player_season_data(self, league_name: str, season_year: str) -> List[Dict]:
        """
        Scrapes aggregate player stats for a season.
        """
        url = f"{self.BASE_URL}/{league_name}/{season_year}"
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            scripts = soup.find_all('script')
            
            for script in scripts:
                if 'playersData' in script.text:
                    json_string = script.text.split("JSON.parse('")[1].split("');")[0]
                    json_string = json_string.encode('utf8').decode('unicode_escape')
                    return json.loads(json_string)
            return []
        except Exception as e:
            logger.error(f"Error scraping players: {e}")
            return []