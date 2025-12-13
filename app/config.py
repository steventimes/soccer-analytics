import os

# Competitions
COMPETITIONS_MAP = {
    "CL": 2001,  # Champions League
    "PL": 2021,  # Premier League
    "PPL": 2017, # Primeira Liga
    "DED": 2003, # Eredivisie
    "BL1": 2002, # Bundesliga
    "FL1": 2015, # Ligue 1
    "SA": 2019,  # Serie A
    "PD": 2014,  # La Liga
    "BSA": 2013, # Série A (Brazil)
    "ELC": 2016, # Championship
    "EC": 2018,  # European Championship
    "WC": 2000   # World Cup
}

# Understat League Mapping
UNDERSTAT_LEAGUE_MAP = {
    "PL": "EPL",      # Premier League -> English Premier League
    "PD": "La_liga",  # La Liga
    "BL1": "Bundesliga",
    "SA": "Serie_A",
    "FL1": "Ligue_1",
    "PPL": "RFPL"     # Russian Premier League
}

# Seasons
SEASONS = [str(x) for x in range(2021, 2025)]
TRAINING_SEASONS = [str(x) for x in range(2021, 2025)]