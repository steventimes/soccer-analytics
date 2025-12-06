import logging
from app.ml_service.train import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_nightly.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

TARGETS = [
    {"name": "Champions League", "id": 2001},
    {"name": "Premier League", "id": 2021},
    {"name": "Primeira Liga (Portugal)", "id": 2017},
    {"name": "Eredivisie (Netherlands)", "id": 2003},
    {"name": "Bundesliga (Germany)", "id": 2002},
    {"name": "Ligue 1 (France)", "id": 2015},
    {"name": "Serie A (Italy)", "id": 2019},
    {"name": "La Liga (Spain)", "id": 2014},
    {"name": "Série A (Brazil)", "id": 2013},
    {"name": "Championship (England)", "id": 2016},
    {"name": "European Championship", "id": 2018},
    {"name": "FIFA World Cup", "id": 2000}
]

TRAINING_SEASONS = ['2020', '2021', '2022', '2023', '2024'] 

def run():
    logger.info("Starting Overnight Training Job...")
    trainer = ModelTrainer()
    
    success_count = 0
    fail_count = 0

    for target in TARGETS:
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing: {target['name']} (ID: {target['id']})")
        logger.info(f"{'='*40}")
        
        try:
            df = trainer.prepare_dataset(target['id'], TRAINING_SEASONS)
            
            if df.empty:
                logger.warning(f"SKIPPING {target['name']} - No match data found in DB.")
                fail_count += 1
                continue
            
            model = trainer.train(df)
            
            if model:
                safe_name = target['name'].replace(" ", "_").replace("(", "").replace(")", "").lower()
                trainer.save_model(f"{safe_name}_model.pkl")
                success_count += 1
                
        except Exception as e:
            logger.error(f"CRITICAL ERROR training {target['name']}: {e}")
            fail_count += 1
            continue

    logger.info(f"\n{'='*40}")
    logger.info(f"Job Complete.")
    logger.info(f"Models Trained: {success_count}")
    logger.info(f"Failed/Skipped: {fail_count}")
    logger.info(f"{'='*40}")

if __name__ == "__main__":
    run()