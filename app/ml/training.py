import pandas as pd
import numpy as np
import logging
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os

from app.data_service.db_session import get_db_service
from app.ml.feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.fe = FeatureEngineer()
        self.le = LabelEncoder()
        
        self.model = XGBClassifier(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=1 
        )
        
    def prepare_dataset(self, competition_id: int, seasons: list) -> pd.DataFrame:
        """
        Fetches matches from DB and transforms them into the format 
        expected by the vectorized FeatureEngineer.
        """
        logger.info(f"--- Building Dataset for Comp ID: {competition_id} ---")
        
        with get_db_service() as service:
            all_matches = []
            for season in seasons:
                matches = service.matches.get_by_competition(competition_id, season)
                logger.info(f"Loaded {len(matches)} matches for season {season}")
                all_matches.extend(matches)
            
            if not all_matches:
                return pd.DataFrame()

            data = []
            for m in all_matches:

                if m.status != 'FINISHED' or m.score_home is None:
                    continue

                row = {
                    'id': m.id,
                    'date': m.utc_date,
                    'season': m.season_year,
                    'home_team': m.home_team_id,
                    'away_team': m.away_team_id,
                    'teamID': m.home_team_id,
                    'opponentID': m.away_team_id,
                    'location': 'h',
                    'result': 'W' if m.winner == 'HOME_TEAM' else ('L' if m.winner == 'AWAY_TEAM' else 'D'),
                    'goals': m.score_home,
                    'xGoals': m.home_xg if m.home_xg is not None else 0.0,
                    
                    'odds_home': m.odds_home,
                    'odds_draw': m.odds_draw,
                    'odds_away': m.odds_away
                }

                row_away = {
                    'id': m.id,
                    'date': m.utc_date,
                    'season': m.season_year,
                    'home_team': m.home_team_id,
                    'away_team': m.away_team_id,
                    'teamID': m.away_team_id,
                    'opponentID': m.home_team_id,
                    'location': 'a',
                    'result': 'W' if m.winner == 'AWAY_TEAM' else ('L' if m.winner == 'HOME_TEAM' else 'D'),
                    'goals': m.score_away,
                    'xGoals': m.away_xg if m.away_xg is not None else 0.0,
                    
                    'odds_home': m.odds_home,
                    'odds_draw': m.odds_draw,
                    'odds_away': m.odds_away
                }
                
                data.append(row)
                data.append(row_away)

            df = pd.DataFrame(data)

            if df.empty: return df
            
            processed_df = self.fe.calculate_rolling_features(df)
            
            return processed_df

    def train(self, df: pd.DataFrame, tune=False):
        if df.empty:
            logger.warning("Dataset is empty. Skipping training.")
            return None

        features = self.fe.features
        missing = [c for c in features if c not in df.columns]
        if missing:
            logger.error(f"Missing features in dataframe: {missing}")
            return None

        X = df[features]
        y = df['target']
        
        logger.info(f"Training on {len(X)} rows with features: {features}")

        tscv = TimeSeriesSplit(n_splits=3)
        
        fold = 1
        scores = []
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            self.model.fit(X_train, y_train)
            preds = self.model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            scores.append(acc)
            logger.info(f"Fold {fold} Accuracy: {acc:.2%}")
            fold += 1
            
        logger.info(f"Average Cross-Val Accuracy: {np.mean(scores):.2%}")
        
        self.model.fit(X, y)
        return self.model

    def save_model(self, name: str):
        if not os.path.exists("models"):
            os.makedirs("models")
        joblib.dump(self.model, f"models/{name}.joblib")
        logger.info(f"Model saved to models/{name}.joblib")