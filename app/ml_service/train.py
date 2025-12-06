import pandas as pd
import numpy as np
import logging
import joblib
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

from app.data_service.data_service_factory import get
from app.data_service.data_type import type_db_data as type
from app.ml_service.feature import feature

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.feature_engine = feature()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.le = LabelEncoder()
        
    def prepare_dataset(self, competition_id: int, seasons: list) -> pd.DataFrame:
        logger.info(f"--- Building Dataset for Comp ID: {competition_id} ---")
        
        all_matches = []
        for season in seasons:
            matches = get(type.COMPETITION_MATCHES, {
                'competition_id': competition_id, 
                'season': season
            })
            if matches:
                all_matches.extend(matches)
            
        logger.info(f"Found {len(all_matches)} total matches. Calculating features...")
        
        data = []
        count = 0
        
        for match in all_matches:
            if match['status'] != 'FINISHED':
                continue
            
            home_id = match['home_team']['id']
            away_id = match['away_team']['id']
            match_date_str = match['utc_date']
            season_yr = match['season']
            
            try:
                features = self.feature_engine.create_match_feature(
                    home_id=home_id,
                    away_id=away_id,
                    competition_id=competition_id,
                    season_yr=season_yr,
                    match_date=match_date_str
                )
                
                winner = match['score']['winner']
                
                row = features.copy()
                row['target'] = winner
                row['date'] = match_date_str
                row['home_team'] = match['home_team']['name']
                row['away_team'] = match['away_team']['name']
                
                data.append(row)
            except Exception as e:
                logger.warning(f"Skipping match {match['id']} due to error: {e}")

            count += 1
            if count % 100 == 0:
                logger.info(f"Processed {count} matches...")

        df = pd.DataFrame(data)
        
        if not df.empty:
            df = df.sort_values('date')
            
        return df

    def train(self, df: pd.DataFrame):
        if df.empty:
            logger.error("No data available to train!")
            return None

        df['target_encoded'] = self.le.fit_transform(df['target'])
        
        drop_cols = ['target', 'target_encoded', 'date', 'home_team', 'away_team']
        X = df.drop(columns=drop_cols, errors='ignore')
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

    def save_model(self, filename="football_model.pkl"):
        os.makedirs("app/models", exist_ok=True)
        path = f"app/models/{filename}"
        joblib.dump(self.model, path)
        joblib.dump(self.le, path.replace(".pkl", "_label_encoder.pkl"))
        logger.info(f"Model saved to {path}")