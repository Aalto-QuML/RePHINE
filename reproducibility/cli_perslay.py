import argparse
from argparse import Namespace


def get_perslay_args() -> Namespace:
    parser = argparse.ArgumentParser(description="PyTorch Geometric.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--logdir", type=str, default="results_perslay", help="Log directory"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="NCI109",
        choices=[
            "PROTEINS",
            "NCI109",
            "NCI1",
            "IMDB-BINARY"
        ],
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument(
        "--max_epochs", type=int, default=300, help="Number of epochs to train."
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Interval for printing train statistics.",
    )
    parser.add_argument("--early_stop_patience", type=int, default=30)
    parser.add_argument("--lr_decay_patience", type=int, default=10)

    return parser.parse_args()
