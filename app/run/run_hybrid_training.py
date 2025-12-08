from app.ml.training_hybrid import HybridTrainer

if __name__ == "__main__":
    print("Starting Hybrid Training Protocol...")
    trainer = HybridTrainer()
    trainer.train_and_evaluate()
    print("Done.")