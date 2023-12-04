import os
import json


def save_setup(args):
    args.logdir = (
        f"{args.logdir}/{args.dataset}/{args.depth}/{args.num_filtrations}/"
        f"{args.filtration_hidden}_hidden/{args.out_dim}_outdim/{args.dim1}_dim1"
    )

    if not os.path.exists(f"{args.logdir}"):
        os.makedirs(f"{args.logdir}")

    with open(f"{args.logdir}/summary.json", "w") as f:
        json.dump(args.__dict__, f, indent=2)
