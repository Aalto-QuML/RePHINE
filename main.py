import torch

from reproducibility.cli_main import get_args
from reproducibility.save_setup import save_setup
from runners.run_main import run_main

if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Save Setup
    save_setup(args)
    results = run_main(args, device)
    torch.save(
        results, f"{args.logdir}/{args.diagram_type}_{args.gnn}_{args.seed}.results"
    )
