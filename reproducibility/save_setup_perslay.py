import json
import os


def save_setup_perslay(args):
    args.logdir = f"{args.logdir}/{args.dataset}/"

    if not os.path.exists(f"{args.logdir}"):
        os.makedirs(f"{args.logdir}")

    with open(f"{args.logdir}/summary.json", "w") as f:
        json.dump(args.__dict__, f, indent=2)
