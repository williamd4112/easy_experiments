
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    print(f"Dataset: {args.dataset}; Model: {args.model}, lr={args.lr}, debug={args.debug}")