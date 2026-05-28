import argparse
from flalgorithm.fedavg import Fedavg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='lastmf')
    parser.add_argument('--output', type=str, default='lastfm')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--data_path', type=str, default='./datasets/lastfm')
    parser.add_argument('--client_num', type=int, default=100)
    parser.add_argument('--model', type=str, default='MF')
    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--early_stop_num', type=int, default=10)
    parser.add_argument('--update_frequency', type=int, default=1)
    parser.add_argument('--agg_lr', type=float, default=0.1)
    args= parser.parse_args()

    Testfed = Fedavg(args)
    Testfed.fit()