import pandas as pd
import numpy as np
import joblib
import logging
from app.ml.feature_engineering import FeatureEngineer
from app.ml.training import ModelTrainer
from app.config import COMPETITIONS_MAP, TRAINING_SEASONS

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class BettingSimulator:
    def __init__(self):
        self.fe = FeatureEngineer()
        self.trainer = ModelTrainer()
        self.bankroll = 1000
        self.unit_size = 50
        self.threshold = 0.05

    def run_simulation(self):
        logger.info("--- Starting Betting Simulation (DB Data) ---")

        all_dfs = []
        for code, comp_id in COMPETITIONS_MAP.items():
            logger.info(f"Loading data for {code}...")

            df = self.trainer.prepare_dataset(comp_id, TRAINING_SEASONS)
            if not df.empty:
                df['competition_code'] = code
                all_dfs.append(df)
        
        if not all_dfs:
            logger.error("No data found in Database.")
            return

        full_df = pd.concat(all_dfs)
        
        processed_df = self.fe.calculate_rolling_features(full_df).dropna()
        
        split_idx = int(len(processed_df) * 0.8)
        test_df = processed_df.iloc[split_idx:].copy()
        
        current_bankroll = self.bankroll
        bets_placed = 0
        wins = 0

        logger.info(f"\nSimulating on {len(test_df)} matches...")
        
        for i, row in test_df.iterrows():
            comp_code = row['competition_code']
            
            model_path = f"models/{comp_code.lower()}_model.joblib"
            try:
                model = joblib.load(model_path)
            except FileNotFoundError:
                continue

            X_input = pd.DataFrame([row])[self.fe.features]
            
            probs = model.predict_proba(X_input)[0]
            p_home_win = probs[2]

            if 'odds_home' in row and row['odds_home'] > 0:
                odds = row['odds_home']
                edge = (p_home_win * odds) - 1

                if edge > self.threshold:
                    bets_placed += 1
                    current_bankroll -= self.unit_size
                    
                    if row['target'] == 2:
                        current_bankroll += (self.unit_size * odds)
                        wins += 1

        roi = ((current_bankroll - self.bankroll) / self.bankroll) * 100
        logger.info(f"\n--- Final Results ---")
        logger.info(f"Bankroll: ${current_bankroll:.2f} (Start: ${self.bankroll})")
        logger.info(f"ROI: {roi:.2f}%")
        logger.info(f"Bets Placed: {bets_placed}")