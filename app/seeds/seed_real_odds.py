import pandas as pd
import requests
import io

import os
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


CSV_SOURCES = {
    2021: "E0", # Premier League
    2016: "E1", # Championship
    2002: "D1", # Bundesliga
    2019: "I1", # Serie A
    2014: "SP1", # La Liga
    2015: "F1", # Ligue 1
    2003: "N1", # Eredivisie
    2017: "P1", # Portugal Liga
}

SEASONS = ["2324", "2425"]
BASE_URL = "https://www.football-data.co.uk/mmz4281/{}/{}.csv"

def get_db_connection():
    """Establishes a connection to the Docker PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host=os.environ.get('DB_HOST', 'football_db'),
            database=os.environ.get('DB_NAME', 'football'),
            user=os.environ.get('DB_USER', 'postgres'),
            password=os.environ.get('DB_PASSWORD', 'football')
        )
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        raise e

def normalize_name(name):
    """Simple normalizer to help match team names"""
    if not isinstance(name, str): return ""
    return name.lower().replace(" fc", "").replace("cf ", "").replace(" ac", "").strip()

def run_seed():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    logger.info("--- STARTING REAL ODDS UPDATE ---")
    
    total_updated = 0
    
    for comp_id, code in CSV_SOURCES.items():
        for season in SEASONS:
            url = BASE_URL.format(season, code)
            logger.info(f"Fetching {code} ({season}) from {url}...")
            
            try:
                response = requests.get(url)
                if response.status_code != 200:
                    logger.warning(f"  Failed to download: {url}")
                    continue

                s = response.content
                df = pd.read_csv(io.StringIO(s.decode('utf-8', errors='ignore')))
                
                # Check for B365 odds
                if 'B365H' not in df.columns:
                    logger.warning(f"  No B365 odds found in {code}-{season}")
                    continue

                df = df[['Date', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A']].dropna()
                
                matches_updated = 0
                
                for _, row in df.iterrows():
                    # Parse Date
                    try:
                        date_str = row['Date']
                        if len(date_str.split("/")[-1]) == 2:
                            date_fmt = "%d/%m/%y"
                        else:
                            date_fmt = "%d/%m/%Y"
                        date_obj = datetime.strptime(date_str, date_fmt).date()
                    except Exception:
                        continue
                            
                    home_clean = normalize_name(row['HomeTeam'])
                    
                    query = """
                        UPDATE matches 
                        SET odds_home = %s, odds_draw = %s, odds_away = %s
                        WHERE date BETWEEN %s AND %s 
                        AND competition_id = %s
                        AND (LOWER(home_team) LIKE %s OR LOWER(home_team) LIKE %s)
                        AND (odds_home = 0 OR odds_home IS NULL)
                    """
                    
                    fuzzy_name = f"{home_clean[:4]}%" 
                    exact_name = f"%{home_clean}%"
                    
                    cursor.execute(query, (
                        row['B365H'], row['B365D'], row['B365A'],
                        date_obj - timedelta(days=1), date_obj + timedelta(days=1),
                        comp_id,
                        exact_name, fuzzy_name
                    ))
                    
                    if cursor.rowcount > 0:
                        matches_updated += cursor.rowcount
                
                conn.commit()
                logger.info(f"  Updated {matches_updated} matches for {code}-{season}")
                total_updated += matches_updated
                
            except Exception as e:
                logger.error(f"  Failed to process {code}-{season}: {e}")

    logger.info("-----------------------------------------------------")
    logger.info(f"TOTAL MATCHES UPDATED WITH REAL ODDS: {total_updated}")
    logger.info("-----------------------------------------------------")
    conn.close()

if __name__ == "__main__":
    run_seed()