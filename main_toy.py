from pathlib import Path

import torch

from reproducibility.cli_toy import get_args_toy
from reproducibility.utils import set_seeds
from runners.run_toy import run_toy

if __name__ == "__main__":
    args = get_args_toy()

    # get data
    file = f"./datasets/data_toy/{args.setting}_seed-{args.seed_dataset}.dat"
    dataset = torch.load(file)

    # device
    set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(f"{args.logdir}/models").mkdir(parents=True, exist_ok=True)

    results, model = run_toy(args, dataset, device)
    torch.save(
        results,
        f"{args.logdir}/{args.setting}{args.seed_dataset}-dim1{args.dim1}-{args.model}-seed_{args.seed}.pth",
    )
    torch.save(
        model.state_dict(),
        f"{args.logdir}/models/{args.setting}{args.seed_dataset}-dim1{args.dim1}-{args.model}-seed_{args.seed}.model",
    )
