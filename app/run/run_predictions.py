from app.ml.predict_upcoming import UpcomingPredictor

if __name__ == "__main__":
    print("Running Weekend Predictions...")
    pred = UpcomingPredictor()
    pred.predict()
    print("Done.")