import logging
from datetime import datetime, timedelta
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
            
            for code, understat_name in UNDERSTAT_LEAGUE_MAP.items():
                if code not in COMPETITIONS_MAP: 
                    continue
                
                for season in SEASONS:
                    logger.info(f"Syncing Match xG for {code} {season}...")

                    data = self.client.fetch_season_data(understat_name, season)
                    if not data: continue
                    
                    count = 0
                    for m in data:                        
                        m_date = m['datetime'][:10]
                        db_match = session.query(Match).filter(
                            func.to_char(Match.utc_date, 'YYYY-MM-DD') == m_date,
                            Match.status == 'FINISHED',
                        ).first()
                        
                        if db_match:
                            db_match.home_xg = float(m['xG']['h'])
                            db_match.away_xg = float(m['xG']['a'])
                            count += 1
                            
                    session.commit()
                    logger.info(f"  -> Updated {count} matches with xG.")

    def sync_players(self):
        with get_db_service() as service:
            session = service.session
            all_teams = session.query(Team).all()
            team_map = {t.name: t.id for t in all_teams}

            for code, understat_name in UNDERSTAT_LEAGUE_MAP.items():
                if code not in COMPETITIONS_MAP: continue
                
                for season in SEASONS:
                    logger.info(f"Syncing Players for {code} {season}...")
                    players_data = self.client.fetch_player_season_data(understat_name, season)
                    
                    form_entries = []
                    for p in players_data:
                        match = process.extractOne(p['team_title'], team_map.keys())
                        if match and match[1] > 80:
                            t_id = team_map[match[0]]
                        else:
                            continue
                        
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

if __name__ == "__main__":
    seeder = UnderstatSeeder()
    seeder.sync_matches()
    seeder.sync_players()