import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from rapidfuzz import process
from app.ml.feature_engineering import FeatureEngineer
from app.data_service.db_session import get_db_service

class HybridPredictor:
    def __init__(self, model_path="models/hybrid_model.joblib"):
        self.model = joblib.load(model_path)
        self.fe = FeatureEngineer()

        self.csv_teams = pd.read_csv("teams.csv")
        self.csv_team_map = {row['name']: row['teamID'] for _, row in self.csv_teams.iterrows()}

    def _get_mapped_csv_id(self, api_team_name):
        """Fuzzy matches API name to CSV name to get ID."""
        match = process.extractOne(api_team_name, self.csv_team_map.keys())
        if match and match[1] > 80: # 80% confidence threshold
            return self.csv_team_map[match[0]]
        return None

    def predict_next_match(self, home_team_api_id, away_team_api_id):
        with get_db_service() as service:
            home_team = service.teams.get_by_id(home_team_api_id)
            away_team = service.teams.get_by_id(away_team_api_id)
            
            if not home_team or not away_team:
                return "Teams not found in DB"

            h_matches = service.matches.get_recent_form(home_team_api_id, datetime.now(), limit=10)
            
            input_data = pd.DataFrame([{
                'rolling_goals_scored': h_matches.get('goals_scored', 0) / 5,
                'rolling_goals_conceded': h_matches.get('goals_conceded', 0) / 5,
                'rolling_wins': h_matches.get('wins', 0),
                'rolling_loss': h_matches.get('losses', 0),
                'is_home': 1
            }])

            probs = self.model.predict_proba(input_data)[0]
            
            return {
                "match": f"{home_team.name} vs {away_team.name}",
                "home_win_prob": round(probs[2] * 100, 1),
                "draw_prob": round(probs[1] * 100, 1),
                "away_win_prob": round(probs[0] * 100, 1)
            }