import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import logging
from datetime import datetime

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
        """
        Fetches matches from DB and generates features for ML.
        """
        logger.info(f"Fetching matches for competition {competition_id}...")
        
        all_matches = []
        
        for season in seasons:
            matches = get(type.COMPETITION_MATCHES, {
                'competition_id': competition_id, 
                'season': season
            })
            
            if not matches: # type: ignore
                logger.warning(f"No matches found for season {season}")
                continue
                
            all_matches.extend(matches)
            
        logger.info(f"Processing {len(all_matches)} matches for feature engineering...")
        
        data = []
        count = 0
        
        # calculate features
        for match in all_matches:
            if match['status'] != 'FINISHED':
                continue
            
            home_id = match['home_team']['id']
            away_id = match['away_team']['id']
            match_date_str = match['utc_date']
            season_yr = match['season']

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
            
            count += 1
            if count % 50 == 0:
                logger.info(f"Processed {count} matches...")

        df = pd.DataFrame(data)
        
        if not df.empty:
            df['target_encoded'] = self.le.fit_transform(df['target'])
            
        return df.sort_values('date')

    def train(self, df: pd.DataFrame):
        """
        Train the model using TimeSeriesSplit to avoid data leakage.
        """
        if df.empty:
            logger.error("No data to train on!")
            return

        drop_cols = ['target', 'target_encoded', 'date', 'home_team', 'away_team']
        X = df.drop(columns=drop_cols, errors='ignore')
        y = df['target_encoded']
        
        logger.info(f"Training on features: {list(X.columns)}")

        tscv = TimeSeriesSplit(n_splits=3)
        
        fold = 1
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            self.model.fit(X_train, y_train)
            preds = self.model.predict(X_test)
            
            acc = accuracy_score(y_test, preds)
            logger.info(f"Fold {fold} Accuracy: {acc:.2f}")
            fold += 1
            
        self.model.fit(X, y)
        logger.info("Final model trained on all data.")
        
        return self.model

    def save_model(self, filepath="football_model.pkl"):
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")

if __name__ == "__main__":
    trainer = ModelTrainer()
    df = trainer.prepare_dataset(competition_id=2021, seasons=['2021', '2022', '2023', '2024', '2025'])
    
    if not df.empty:
        print(df.head())
        trainer.train(df)
        trainer.save_model()