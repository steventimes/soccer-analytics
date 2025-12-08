import logging
from datetime import datetime
from rapidfuzz import process
from sqlalchemy import func

from app.data_service.db_session import get_db_service
from app.data_service.fetch.understat_client import UnderstatClient
from app.data_service.db.database.db_schema import Match, Team, PlayerForm, Player
from app.config import COMPETITIONS_MAP, UNDERSTAT_LEAGUE_MAP, SEASONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnderstatSeeder:
    def __init__(self):
        self.client = UnderstatClient()

    def sync_matches(self):
        """
        Fetches xG data from Understat and updates existing matches in DB.
        """
        with get_db_service() as service:
            session = service.session
            all_teams = session.query(Team).all()
            team_map = {t.name: t.id for t in all_teams}
            team_names = list(team_map.keys())

            for code, understat_name in UNDERSTAT_LEAGUE_MAP.items():
                if code not in COMPETITIONS_MAP: continue
                
                db_comp_id = COMPETITIONS_MAP[code]
                
                for season in SEASONS:
                    logger.info(f"Syncing {code} {season}...")
                    
                    data = self.client.fetch_season_data(understat_name, season)
                    if not data: continue
                    
                    updates = 0
                    for m in data:                        
                        if not m['isResult']: continue # Skip unplayed matches

                        h_name = m['h']['title']
                        a_name = m['a']['title']

                        h_match = process.extractOne(h_name, team_names, score_cutoff=80)
                        a_match = process.extractOne(a_name, team_names, score_cutoff=80)
                        
                        if not h_match or not a_match:
                            continue
                            
                        h_id = team_map[h_match[0]]
                        a_id = team_map[a_match[0]]
                        
                        match_date = datetime.strptime(m['datetime'], "%Y-%m-%d %H:%M:%S")

                        db_match = session.query(Match).filter(
                            Match.home_team_id == h_id,
                            Match.away_team_id == a_id,
                            Match.season_year == str(season),
                            func.abs(func.date(Match.date) - match_date.date()) <= 1
                        ).first()
                        
                        if db_match:
                            db_match.home_xg = float(m['xG']['h'])
                            db_match.away_xg = float(m['xG']['a'])
                            updates += 1
                    
                    session.commit()
                    logger.info(f"  -> Updated {updates} matches with xG.")

    def sync_players(self):
        """
        Fetches seasonal player stats (xG, xA) from Understat.
        """
        with get_db_service() as service:
            session = service.session

            teams_map = {t.name: t.id for t in session.query(Team).all()}
            team_names = list(teams_map.keys())

            for code, understat_name in UNDERSTAT_LEAGUE_MAP.items():
                for season in SEASONS:
                    logger.info(f"Fetching Players: {code} {season}...")
                    data = self.client.fetch_player_season_data(understat_name, season)
                    
                    form_entries = []
                    for p in data:
                        t_match = process.extractOne(p['team_title'], team_names, score_cutoff=80)
                        t_id = teams_map[t_match[0]] if t_match else None
                        
                        entry = PlayerForm(
                            period_label=f"{season}_season",
                            team_id=t_id,
                            goals=int(p['goals']),
                            xg=float(p['xG']),
                            assists=int(p['assists']),
                            xa=float(p['xA']),
                            shots=int(p['shots']),
                            key_passes=int(p['key_passes']),
                            yellow_cards=int(p['yellow_cards']),
                            red_cards=int(p['red_cards']),
                            npg=float(p['npg']),
                            npxg=float(p['npxG']),
                            xg_chain=float(p['xGChain']),
                            xg_buildup=float(p['xGBuildup']),
                            minutes_played=int(p['time']),
                            games_played=int(p['games'])
                        )
                        form_entries.append(entry)

                    session.query(PlayerForm).filter_by(period_label=f"{season}_season").delete()                    
                    session.add_all(form_entries)
                    session.commit()
                    logger.info(f"  -> Saved {len(form_entries)} player records.")

if __name__ == "__main__":
    seeder = UnderstatSeeder()
    seeder.sync_matches()
    seeder.sync_players()