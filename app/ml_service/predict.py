import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional

from app.data_service.data_service_factory import get
from app.data_service.data_type import type_db_data as data_type
from app.ml_service.feature import feature

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger.getLogger(__name__)

class MatchPredictor:
    def __init__(self, model_path: str = None): # type: ignore
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.feature_engine = feature()
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """
        Load trained model and artifacts
        """
        try:
            self.model = joblib.load(f"{model_path}.joblib")

            self.scaler = joblib.load(f"{model_path}_scaler.joblib")

            self.feature_names = joblib.load(f"{model_path}_features.joblib")
            
            logger.info(f"Model loaded successfully from {model_path}")
            logger.info(f"Model type: {type(self.model).__name__}")
            logger.info(f"Number of features: {len(self.feature_names)}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict_match(
        self,
        home_team_id: int,
        away_team_id: int,
        competition_id: int,
        season_year: str,
        match_date: Optional[datetime] = None
    ) -> Dict:
        """
        Predict outcome for a single match
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        features = self.feature_engine.create_match_feature(
            home_id=home_team_id,
            away_id=away_team_id,
            competition_id=competition_id,
            season_yr=season_year,
            match_date=match_date
        )

        X = pd.DataFrame([features])[self.feature_names]

        X_scaled = self.scaler.transform(X) # type: ignore

        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]

        class_labels = self.model.classes_
        prob_dict = {label: prob for label, prob in zip(class_labels, probabilities)}
        
        return {
            'prediction': prediction,
            'probabilities': prob_dict,
            'features': features,
            'confidence': max(probabilities)
        }
    
    def predict_multiple_matches(self, matches: List[Dict]) -> pd.DataFrame:
        """
        Predict outcomes for multiple matches
        """
        results = []
        
        for match in matches:
            try:
                prediction = self.predict_match(
                    home_team_id=match['home_team_id'],
                    away_team_id=match['away_team_id'],
                    competition_id=match['competition_id'],
                    season_year=match['season_year'],
                    match_date=match.get('match_date')
                )
                
                result = {
                    'match_id': match.get('match_id'),
                    'home_team_id': match['home_team_id'],
                    'away_team_id': match['away_team_id'],
                    'predicted_result': prediction['prediction'],
                    'confidence': prediction['confidence'],
                    'prob_home_win': prediction['probabilities'].get('H', 0),
                    'prob_draw': prediction['probabilities'].get('D', 0),
                    'prob_away_win': prediction['probabilities'].get('A', 0),
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to predict match {match.get('match_id')}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def get_prediction_breakdown(
        self,
        home_team_id: int,
        away_team_id: int,
        competition_id: int,
        season_year: str
    ) -> Dict:
        """
        Get detailed prediction breakdown with feature contributions
        """
        prediction = self.predict_match(
            home_team_id, away_team_id, competition_id, season_year
        )

        home_team = get(data_type.SINGLE_TEAM, home_team_id)
        away_team = get(data_type.SINGLE_TEAM, away_team_id)
        
        home_team_name = home_team.iloc[0]['name'] if not home_team.empty else f"Team {home_team_id}"
        away_team_name = away_team.iloc[0]['name'] if not away_team.empty else f"Team {away_team_id}"
        
        breakdown = {
            'match': f"{home_team_name} vs {away_team_name}",
            'prediction': prediction['prediction'],
            'confidence': prediction['confidence'],
            'probabilities': prediction['probabilities'],
            'key_factors': self._analyze_key_factors(prediction['features'])
        }
        
        return breakdown
    
    def _analyze_key_factors(self, features: Dict) -> List[Dict]:
        """
        Analyze which features most influenced the prediction
        """
        if not hasattr(self.model, 'feature_importances_'):
            return []

        importance_scores = dict(zip(self.feature_names, self.model.feature_importances_)) # type: ignore

        key_factors = []
        for feature_name, value in features.items():
            if feature_name in importance_scores:
                key_factors.append({
                    'feature': feature_name,
                    'value': value,
                    'importance': importance_scores[feature_name]
                })

        key_factors.sort(key=lambda x: x['importance'], reverse=True)
        return key_factors[:5]
    
    def predict_upcoming_matches(
        self,
        competition_code: str = 'PL',
        season_year: str = None # type: ignore
    ) -> pd.DataFrame:
        """
        Predict upcoming matches for a competition
        """
        if season_year is None:
            season_year = str(datetime.now().year)

        matches_df = get(data_type.COMPETITION_MATCHES, {
            'competition_code': competition_code,
            'season': season_year
        })
        
        if matches_df.empty:
            logger.warning("No matches found for prediction")
            return pd.DataFrame()

        upcoming_matches = matches_df[matches_df['status'] != 'FINISHED']
        
        if upcoming_matches.empty:
            logger.warning("No upcoming matches found")
            return pd.DataFrame()
        
        logger.info(f"Found {len(upcoming_matches)} upcoming matches")

        matches_list = []
        for _, match in upcoming_matches.iterrows():
            matches_list.append({
                'match_id': match['id'],
                'home_team_id': match['home_team_id'],
                'away_team_id': match['away_team_id'],
                'competition_id': match['competition_id'],
                'season_year': season_year,
                'match_date': match.get('match_date')
            })
        
        return self.predict_multiple_matches(matches_list)

def main():
    """
    Main prediction function - example usage
    """
    logger.info("Starting match prediction...")

    predictor = MatchPredictor()
    
    try:
        model_path = "app/ml/models/football_prediction_model"
        predictor.load_model(model_path)
        
        logger.info("Predicting upcoming Premier League matches...")
        predictions = predictor.predict_upcoming_matches(competition_code='PL')
        
        if not predictions.empty:
            print("\n" + "="*80)
            print("UPCOMING MATCH PREDICTIONS")
            print("="*80)
            
            for _, pred in predictions.iterrows():
                home_team = get(data_type.SINGLE_TEAM, pred['home_team_id'])
                away_team = get(data_type.SINGLE_TEAM, pred['away_team_id'])
                
                home_name = home_team.iloc[0]['name'] if not home_team.empty else f"Team {pred['home_team_id']}"
                away_name = away_team.iloc[0]['name'] if not away_team.empty else f"Team {pred['away_team_id']}"
                
                print(f"{home_name} vs {away_name}")
                print(f"  Prediction: {pred['predicted_result']}")
                print(f"  Confidence: {pred['confidence']:.1%}")
                print(f"  Probabilities - Home: {pred['prob_home_win']:.1%}, "
                      f"Draw: {pred['prob_draw']:.1%}, Away: {pred['prob_away_win']:.1%}")
                print("-" * 50)
        
        logger.info("\nSingle match prediction with breakdown...")
        breakdown = predictor.get_prediction_breakdown(
            home_team_id=66,
            away_team_id=64,
            competition_id=2021,
            season_year="2024"
        )
        
        print("\n" + "="*80)
        print("DETAILED PREDICTION BREAKDOWN")
        print("="*80)
        print(f"Match: {breakdown['match']}")
        print(f"Prediction: {breakdown['prediction']}")
        print(f"Confidence: {breakdown['confidence']:.1%}")
        print(f"Probabilities: {breakdown['probabilities']}")
        
        print("\nKey Factors:")
        for factor in breakdown['key_factors']:
            print(f"  - {factor['feature']}: {factor['value']} (importance: {factor['importance']:.3f})")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise

if __name__ == "__main__":
    main()