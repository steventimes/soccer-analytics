import joblib
import logging
import pandas as pd
from app.ml.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

class MatchPredictor:
    def __init__(self, model_path: str = None):
        self.model = None
        self.le = None
        self.feature_engine = FeatureEngineer()
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        try:
            self.model = joblib.load(f"{model_path}.joblib")
            self.le = joblib.load(f"{model_path}_le.joblib")
            logger.info(f"Model loaded: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def predict_match(self, home_team_id: int, away_team_id: int, date, service):
        """
        Predict a single match outcome.
        Requires a simpler 'Fake Match' object or restructuring FeatureEngineer to accept IDs.
        For now, we mock a match object.
        """
        class MockMatch:
            def __init__(self, h, a, d):
                self.home_team_id = h
                self.away_team_id = a
                self.utc_date = d
                self.status = 'SCHEDULED'
                
        mock_match = MockMatch(home_team_id, away_team_id, date)
        
        features = self.feature_engine.calculate_features(mock_match, service) # type: ignore
        
        df = pd.DataFrame([features])

        if self.model and self.le:
            pred_idx = self.model.predict(df)[0]
            pred_label = self.le.inverse_transform([pred_idx])[0]
            probs = self.model.predict_proba(df)[0]
            
            return {
                'prediction': pred_label,
                'confidence': max(probs),
                'probabilities': dict(zip(self.le.classes_, probs))
            }
        return None