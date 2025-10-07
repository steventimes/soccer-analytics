import os
import requests
from dotenv import load_dotenv
import pandas as pd

load_dotenv()  # load API key
API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")

BASE_URL = "https://api.football-data.org/v4/"


def get_team_players(team_id: int):
    """Fetch players from a specific team"""
    url = f"{BASE_URL}teams/{team_id}"
    headers = {"X-Auth-Token": API_KEY}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()

    players = data.get("squad", [])
    df = pd.DataFrame(players)
    return df


def get_competition_standings(competition_code="PL"):
    """Fetch Premier League (default) standings"""
    url = f"{BASE_URL}competitions/{competition_code}/standings"
    headers = {"X-Auth-Token": API_KEY}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()

    standings = data["standings"][0]["table"]
    df = pd.DataFrame(standings)
    return df


if __name__ == "__main__":
    df_players = get_team_players(66)  # Man United
    print(df_players[["name", "position", "nationality"]].head())

    df_standings = get_competition_standings("PL")
    print(df_standings[["position", "team", "points"]].head())
