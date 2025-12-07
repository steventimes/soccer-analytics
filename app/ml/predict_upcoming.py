import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime, timedelta
from app.data_service.fetch.fetcher import FootballDataClient
from app.data_service.db_session import get_db_service
from app.ml.feature_engineering import FeatureEngineer
from app.config import COMPETITIONS_MAP

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class UpcomingPredictor:
    def __init__(self):
        self.client = FootballDataClient()
        self.fe = FeatureEngineer()
        self.model_path = "models/hybrid_model.joblib"
        try:
            self.model = joblib.load(self.model_path)
        except FileNotFoundError:
            logger.error("Model not found. Train it first!")
            self.model = None

    def get_upcoming_matches(self, days=3):
        """Fetch scheduled matches for supported competitions."""
        upcoming = []
        date_from = datetime.now().strftime("%Y-%m-%d")
        date_to = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
        
        for code, comp_id in COMPETITIONS_MAP.items():
            try:
                matches = self.client._get(f"competitions/{code}/matches", {
                    "status": "SCHEDULED",
                    "dateFrom": date_from,
                    "dateTo": date_to
                })
                
                if matches and 'matches' in matches:
                    for m in matches['matches']:
                        upcoming.append({
                            'id': m['id'],
                            'comp_code': code,
                            'utc_date': m['utcDate'],
                            'home_team': m['homeTeam']['name'],
                            'home_id': m['homeTeam']['id'],
                            'away_team': m['awayTeam']['name'],
                            'away_id': m['awayTeam']['id']
                        })
            except Exception as e:
                logger.error(f"Error fetching {code}: {e}")
                
        return upcoming

    def predict(self):
        if not self.model: return

        matches = self.get_upcoming_matches()
        if not matches:
            logger.info("No upcoming matches found in the next 3 days.")
            return

        logger.info(f"--- PREDICTING {len(matches)} UPCOMING MATCHES ---")
        
        with get_db_service() as service:
            for m in matches:
                h_history = service.matches.get_recent_form(m['home_id'], datetime.now(), limit=10)
                a_history = service.matches.get_recent_form(m['away_id'], datetime.now(), limit=10)
                
                input_data = {
                    'rolling_xG': 0,
                    'rolling_xGA': 0,
                    'rolling_deep': 0,
                    'rolling_ppda': 20.0,
                    'rolling_goals': h_history.get('goals_scored', 0) / 5, 
                    'rolling_wins': h_history.get('wins', 0),
                    'is_home': 1,
                    'xG_diff': 0,
                    'ppda_diff': 0,
                    'deep_diff': 0,
                    'points_diff': abs((h_history.get('wins',0)*3 + h_history.get('draws',0)) - 
                                       (a_history.get('wins',0)*3 + a_history.get('draws',0)))
                }
                
                df = pd.DataFrame([input_data])
                
                for feature in self.fe.features:
                    if feature not in df.columns:
                        df[feature] = 0
  
                X = df[self.fe.features]

                probs = self.model.predict_proba(X)[0]

                p_loss, p_draw, p_win = probs[0], probs[1], probs[2]

                if p_win > 0.45: prediction = "HOME WIN"
                elif p_loss > 0.45: prediction = "AWAY WIN"
                else: prediction = "DRAW (Risk)"

                confidence = max(p_win, p_loss, p_draw) * 100
                
                logger.info(f"\n {m['home_team']} vs {m['away_team']}")
                logger.info(f"   Prediction: {prediction} ({confidence:.1f}%)")
                logger.info(f"   Probs: Home {p_win:.2f} | Draw {p_draw:.2f} | Away {p_loss:.2f}")

if __name__ == "__main__":
    predictor = UpcomingPredictor()
    predictor.predict()