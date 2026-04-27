import argparse
import importlib
import sys


MODELS = ["clip"]


def parse_base_args():
    parser = argparse.ArgumentParser(
        description="CV model trainer",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="clip",
        choices=MODELS,
        help="Model to train: " + ", ".join(MODELS),
    )
    # Parse only --model; remaining args are forwarded to model-specific runner
    return parser.parse_known_args()


def main():
    base_args, remaining = parse_base_args()

    # Dynamically load models/<name>/train.py and call run()
    module = importlib.import_module(f"models.{base_args.model}.train")
    module.run(remaining)


if __name__ == "__main__":
    main()
