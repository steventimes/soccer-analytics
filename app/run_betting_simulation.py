from app.ml.simulate_betting import BettingSimulator

if __name__ == "__main__":
    print("Starting Betting Simulation...")
    sim = BettingSimulator()
    sim.run_simulation()
    print("Simulation Complete.")