import pandas as pd
import numpy as np
import logging
import joblib
import os
import random
from app.ml.training import ModelTrainer
from app.config import COMPETITIONS_MAP, TRAINING_SEASONS

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def run_simulation():
    trainer = ModelTrainer()
    bankroll = 1000.0
    total_bets = 0
    
    logger.info("--- Starting Verbose Betting Simulation ---")
    
    normalized_comps = []
    for k, v in COMPETITIONS_MAP.items():
        if str(k).isdigit():
            normalized_comps.append((int(k), v))
        elif str(v).isdigit():
            normalized_comps.append((int(v), k))
            
    for comp_id, comp_code in normalized_comps:
        
        model_path = f"models/{comp_code}_model.joblib"
        if not os.path.exists(model_path):
             model_path = f"models/{comp_code.lower()}_model.joblib"
        
        if not os.path.exists(model_path):
            continue

        try:
            model = joblib.load(model_path)
        except Exception as e:
            logger.error(f"Failed to load model for {comp_code}: {e}")
            continue

        logger.info(f"\nAnalyzing {comp_code} (ID: {comp_id})...")
        
        df = trainer.prepare_dataset(comp_id, [2024])
        
        if df.empty:
            logger.warning("  No data found.")
            continue

        avg_odds = df['odds_home'].mean()
        using_synthetic_odds = False
        
        if avg_odds <= 1.01:
            logger.warning(f"  [WARNING] NO ODDS DATA FOUND (Avg: {avg_odds:.2f})")
            logger.warning("  -> ACTIVATING SYNTHETIC ODDS MODE.")
            logger.warning("  (Generating simulated bookmaker odds to test model logic)")
            using_synthetic_odds = True
        else:
            logger.info(f"  Average Home Odds found: {avg_odds:.2f}")

        features = trainer.fe.features
        valid_features = [c for c in features if c in df.columns]
        X = df[valid_features]
        
        try:
            probs = model.predict_proba(X)
            home_win_probs = probs[:, 2] 
            df['prob_home'] = home_win_probs
        except Exception as e:
            logger.error(f"  Prediction error: {e}")
            continue
        
        bets_placed_this_comp = 0
        
        for i, (idx, row) in enumerate(df.iterrows()):
            prob = row['prob_home']
            odds = row['odds_home']

            if using_synthetic_odds:
                fair_odds = 1.0 / prob if prob > 0 else 1.0
                noise = random.uniform(0.85, 1.15) 
                odds = fair_odds * noise
                odds = max(1.01, min(odds, 25.0))

            edge = (prob * odds) - 1

            if edge > 0.03:
                stake = 50
                
                result = row['result'] # W, D, L
                
                if result == 'W':
                    profit = (stake * odds) - stake
                    bankroll += profit
                    outcome = "WIN "
                else:
                    bankroll -= stake
                    profit = -stake
                    outcome = "LOSS"
                
                total_bets += 1
                bets_placed_this_comp += 1
                
                if bets_placed_this_comp <= 3:
                    real_or_fake = "[SYNTH]" if using_synthetic_odds else "[REAL]"
                    logger.info(f"  BET {real_or_fake}: Edge {edge:.2f} | Prob {prob:.2f} vs Odds {odds:.2f} | Result: {result} -> {outcome} (${profit:.2f})")
            
            elif i < 3: 
                 logger.info(f"  SKIP: Edge {edge:.2f} (Prob {prob:.2f} * Odds {odds:.2f}) is too low.")

    logger.info("\n===============================")
    logger.info(f"Final Bankroll: ${bankroll:.2f}")
    logger.info(f"Total Bets: {total_bets}")
    logger.info(f"ROI: {((bankroll - 1000)/1000)*100:.2f}%")
    logger.info("===============================")

if __name__ == "__main__":
    run_simulation()