import logging
import time

from app.ml.training import ModelTrainer
from app.config import COMPETITIONS_MAP, TRAINING_SEASONS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run():
    logger.info("Starting Training Pipeline...")
    trainer = ModelTrainer()
    
    success_count = 0
    fail_count = 0

    for code, comp_id in COMPETITIONS_MAP.items():
        logger.info(f"\n{'='*40}")
        logger.info(f"Training Model for: {code} (ID: {comp_id})")
        logger.info(f"{'='*40}")
        
        try:
            df = trainer.prepare_dataset(comp_id, TRAINING_SEASONS)
            
            if df.empty:
                logger.warning(f"SKIPPING {code} - No data found.")
                fail_count += 1
                continue
            
            model = trainer.train(df)
            
            if model:
                safe_name = f"{code.lower()}_model"
                trainer.save_model(safe_name)
                success_count += 1
                
        except Exception as e:
            logger.error(f"Training failed for {code}: {e}")
            fail_count += 1

    logger.info(f"\nPipeline Complete. Success: {success_count}, Failed: {fail_count}")

if __name__ == "__main__":
    run()