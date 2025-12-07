import pandas as pd
import numpy as np
import logging
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from app.data_service.db_session import get_db_service
from app.ml.feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.feature_engine = FeatureEngineer()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.le = LabelEncoder()
        
    def prepare_dataset(self, competition_id: int, seasons: list) -> pd.DataFrame:
        """
        Fetch matches from DB and build feature set.
        """
        logger.info(f"--- Building Dataset for Comp ID: {competition_id} ---")
        
        data = []
 
        with get_db_service() as service:
            all_matches = []
            
            for season in seasons:
                matches = service.matches.get_by_competition(competition_id, season)
                if matches:
                    all_matches.extend(matches)
            
            logger.info(f"Found {len(all_matches)} total matches. Calculating features...")
            
            count = 0
            for match in all_matches:
                if match.status != 'FINISHED':
                    continue
                
                features = self.feature_engine.calculate_features(match, service)

                row = features.copy()
                row['target'] = match.winner
                row['date'] = match.utc_date
                row['home_team'] = match.home_team_id
                row['away_team'] = match.away_team_id
                
                data.append(row)
                count += 1
                
        logger.info(f"Processed {count} matches with features.")
        return pd.DataFrame(data)

    def train(self, df: pd.DataFrame):
        if df.empty:
            logger.error("No data available to train!")
            return None

        if 'target' not in df.columns or df['target'].isnull().all():
            logger.error("No valid target data found.")
            return None

        df = df.dropna(subset=['target'])
        df['target_encoded'] = self.le.fit_transform(df['target'].astype(str))

        drop_cols = ['target', 'target_encoded', 'date', 'home_team', 'away_team']
        X = df.drop(columns=drop_cols, errors='ignore')
        X = X.select_dtypes(include=[np.number])
        y = df['target_encoded']
        
        logger.info(f"Training Features: {list(X.columns)}")

        tscv = TimeSeriesSplit(n_splits=3)
        fold = 1
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            self.model.fit(X_train, y_train)
            preds = self.model.predict(X_test)
            
            acc = accuracy_score(y_test, preds)
            logger.info(f"Validation Fold {fold} Accuracy: {acc:.2%}")
            fold += 1

        self.model.fit(X, y)
        logger.info("Final Model Trained successfully.")
        
        return self.model

    def save_model(self, filename: str):
        if not os.path.exists("models"):
            os.makedirs("models")
            
        path = os.path.join("models", filename)
        joblib.dump(self.model, f"{path}.joblib")
        joblib.dump(self.le, f"{path}_le.joblib")
        logger.info(f"Model saved to {path}")