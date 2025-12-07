import pandas as pd
import numpy as np
import logging
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from app.data_service.db_session import get_db_service
from app.ml.feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.fe = FeatureEngineer()
        self.le = LabelEncoder()
        
        # XGBoost Configuration
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
                for m in matches:
                    all_matches.append(m.to_dict())

            if not all_matches:
                logger.warning("No matches found in DB.")
                return pd.DataFrame()

            df = pd.DataFrame(all_matches)
            
            if 'utc_date' in df.columns:
                df['date'] = pd.to_datetime(df['utc_date'])
            
            
            home_df = df.copy()
            home_df['teamID'] = home_df['home_team_id']
            home_df['opponentID'] = home_df['away_team_id']
            home_df['location'] = 'h'
            home_df['goals'] = home_df['score_home']
            home_df['xGoals'] = 0 # Placeholder if not available in API data
            
            home_map = {'HOME_TEAM': 'W', 'AWAY_TEAM': 'L', 'DRAW': 'D'}
            home_df['result'] = home_df['winner'].map(home_map)
            
            away_df = df.copy()
            away_df['teamID'] = away_df['away_team_id']
            away_df['opponentID'] = away_df['home_team_id']
            away_df['location'] = 'a'
            away_df['goals'] = away_df['score_away']
            away_df['xGoals'] = 0 # Placeholder
            
            away_map = {'AWAY_TEAM': 'W', 'HOME_TEAM': 'L', 'DRAW': 'D'}
            away_df['result'] = away_df['winner'].map(away_map)
            
            full_df = pd.concat([home_df, away_df], ignore_index=True)
            
            logger.info(f"Calculating rolling features for {len(full_df)} team-match rows...")
            processed_df = self.fe.calculate_rolling_features(full_df)
            
            processed_df = processed_df.dropna()
            
            return processed_df

    def train(self, df: pd.DataFrame, tune=False):
        if df.empty:
            logger.warning("Dataset is empty. Skipping training.")
            return None

        features = self.fe.features
        
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
        import os
        os.makedirs("models", exist_ok=True)
        
        path = f"models/{name}.joblib"
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")