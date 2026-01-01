import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime, timedelta
from app.data_service.fetch.fetcher import FootballDataClient
from app.ml.feature_engineering import FeatureEngineer
from app.config import COMPETITIONS_MAP

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class UpcomingPredictor:
    def __init__(self):
        self.client = FootballDataClient()
        self.fe = FeatureEngineer()

    def predict(self, days=3):
        """Fetch scheduled matches and predict outcomes."""
        date_from = datetime.now().strftime("%Y-%m-%d")
        date_to = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
        
        logger.info(f"Fetching matches from {date_from} to {date_to}...")

        for code, comp_id in COMPETITIONS_MAP.items():
            model_path = f"models/{code.lower()}_model.joblib"
            try:
                model = joblib.load(model_path)
            except FileNotFoundError:
                logger.warning(f"No model found for {code} ({model_path}). Skipping.")
                continue

            matches_data = self.client._get(f"competitions/{code}/matches", {
                "status": "SCHEDULED",
                "dateFrom": date_from,
                "dateTo": date_to
            })
            
            if not matches_data or 'matches' not in matches_data:
                continue
                
            matches = matches_data['matches']
            if not matches:
                continue

            logger.info(f"--- Analyzing {code} ({len(matches)} games) ---")

            for m in matches:
                home_team = m['homeTeam']['name']
                away_team = m['awayTeam']['name']

                input_data = {
                    'rolling_xG': 1.5,
                    'rolling_xGA': 1.2,
                    'rolling_deep': 5,
                    'rolling_ppda': 10,
                    'rolling_goals': 1.2,
                    'rolling_wins': 0.5,
                    'is_home': 1,
                    'xG_diff': 0.1,
                    'ppda_diff': -2,
                    'deep_diff': 1,
                    'points_diff': 5
                }
                
                df = pd.DataFrame([input_data])
                
                X = df[self.fe.features]

                probs = model.predict_proba(X)[0]
                p_loss, p_draw, p_win = probs[0], probs[1], probs[2]

                if p_win > 0.45: 
                    prediction = "HOME WIN"
                elif p_loss > 0.45: 
                    prediction = "AWAY WIN"
                else: 
                    prediction = "DRAW (Risk)"

                confidence = max(p_win, p_loss, p_draw) * 100
                
                logger.info(f"{home_team} vs {away_team}")
                logger.info(f"Pred: {prediction} ({confidence:.1f}%)")
                logger.info(f"Probs: H:{p_win:.2f} D:{p_draw:.2f} A:{p_loss:.2f}\n")