from reproducibility.cli_perslay import get_perslay_args
from reproducibility.save_setup_perslay import save_setup_perslay
from runners.run_perslay import run_perslay
from train import *

if __name__ == "__main__":

    args = get_perslay_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_setup_perslay(args)
    results = run_perslay(args, device)

    torch.save(results, f'{args.logdir}/perslay_{args.seed}.results')