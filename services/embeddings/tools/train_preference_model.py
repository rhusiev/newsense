import argparse
from embeddings.models import run_benchmark, train_all_users

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train user preference models.")
    parser.add_argument("--no-weights", action="store_true", help="Disable class weighting")
    parser.add_argument("--benchmark", action="store_true", help="Run model comparison benchmark")
    args = parser.parse_args()
    
    if args.benchmark:
        run_benchmark()
    else:
        train_all_users(use_weights=not args.no_weights)
