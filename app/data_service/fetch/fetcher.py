import os
import requests
from dotenv import load_dotenv
import pandas as pd
from app.data_service.data_type import type_db_data

# load API key and website
load_dotenv()
API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
BASE_URL = "https://api.football-data.org/v4/"

#function that will be called by other functions
def api_get(type, data) -> pd.DataFrame:
    match type:
        case type_db_data.TEAM_PLAYER:
            return get_team_players(data)
        case type_db_data.COMPETITION_STANDING:
            return get_competition_standings(data)
        case type_db_data.SINGLE_TEAM:
            return pd.DataFrame() #TODO: implement single get team? OR it is included
    return pd.DataFrame()
    
#section below is implementation of get based on different types
def get_team_players(team_id: int):
    """Fetch players from a specific team"""
    url = f"{BASE_URL}teams/{team_id}"
    data = fetch(url)

    players = data.get("squad", [])
    df = pd.DataFrame(players)
    return df


def get_competition_standings(competition_code="PL"):
    """Fetch Premier League (default) standings"""
    url = f"{BASE_URL}competitions/{competition_code}/standings"
    data = fetch(url)
    standings = data["standings"][0]["table"]
    df = pd.DataFrame(standings)
    return df

#get data from website
def fetch(url):
    headers = {"X-Auth-Token": API_KEY}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    return data



#module test function
if __name__ == "__main__":
    df_players = get_team_players(66)  # Man United
    print(df_players[["name", "position", "nationality"]].head())

    df_standings = get_competition_standings("PL")
    print(df_standings[["position", "team", "points"]].head())
