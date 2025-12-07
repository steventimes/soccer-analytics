import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from app.utils.csv_loader import CSVLoader
from app.ml.feature_engineering import FeatureEngineer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridTrainer:
    def __init__(self):
        self.loader = CSVLoader()
        self.fe = FeatureEngineer()
        self.model = XGBClassifier(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=5,
            objective='multi:softprob',
            n_jobs=1
        )

    def train_and_evaluate(self):
        logger.info("Loading Historical CSV Data...")
        raw_df = self.loader.load_and_merge()
        
        if raw_df.empty:
            logger.error("No data found.")
            return

        logger.info("Generating Features...")
        processed_df = self.fe.calculate_rolling_features(raw_df)
        processed_df = processed_df.sort_values('date')
        processed_df = processed_df.dropna()

        sample_weights = np.ones(len(processed_df))
        sample_weights[processed_df['target'] == 1] = 2.0 

        X = processed_df[self.fe.features]
        y = processed_df['target']
        
        split_idx = int(len(processed_df) * 0.8)
        
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        w_train = sample_weights[:split_idx]
        
        logger.info(f"Training on {len(X_train)} games...")
        
        self.model.fit(X_train, y_train, sample_weight=w_train)
        
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        logger.info(f"Model Accuracy: {acc:.2%}")
        
        logger.info("\n" + classification_report(y_test, preds, target_names=['Loss', 'Draw', 'Win']))
        
        logger.info("Retraining on full dataset for production...")
        self.model.fit(X, y, sample_weight=sample_weights)
        joblib.dump(self.model, "models/hybrid_model.joblib")
        logger.info("Final Production Model saved.")

if __name__ == "__main__":
    trainer = HybridTrainer()
    trainer.train_and_evaluate()