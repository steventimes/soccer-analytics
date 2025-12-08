import argparse
from app.utils.csv_ingestor import CSVIngestor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, 
                        choices=['matches', 'players', 'understat_players'])
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--period", type=str, help="Label for player stats (e.g., '2015_season')")
    parser.add_argument("--countries", type=str)

    args = parser.parse_args()
    ingestor = CSVIngestor()
    
    if args.source == 'matches':
        ingestor.ingest_matches(args.file, args.countries)
    elif args.source == 'players':
        ingestor.ingest_players(args.file)
    elif args.source == 'understat_players':
        if not args.period:
            print("Error: --period required (e.g. '2015_season')")
        else:
            ingestor.ingest_understat_players(args.file, args.period)
            
    print("Done!")