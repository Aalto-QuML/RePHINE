import argparse


def get_args_toy():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="./results/toy", help="Log directory")
    parser.add_argument(
        "--setting",
        type=str,
        choices=["cub08-1", "cub10-2", "cub12-3"],
        default="cub10-2",
    )
    parser.add_argument("--seed_dataset", type=int, default=42)

    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--n_filtrations", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no-dim1", dest="dim1", action="store_false")
    parser.add_argument("--reduce_tuples", action="store_true")
    parser.add_argument(
        "--model", type=str, choices=["gcn", "rephine", "standard"], default="rephine"
    )

    return parser.parse_args()
