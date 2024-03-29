import argparse
from dataclasses import dataclass


@dataclass
class Config:
    dataset: str = 'movielens'
    model: str = 'lightgcn'
    seed: int = 100
    min_user_cnt: int = 5
    min_item_cnt: int = 5
    split_method: str = 'random'
    test_ratio: float = 0.2
    val_ratio: float = 0.08
    num_neg_samples: int = 5
    loss: str = 'bpr'
    num_layers: int = 3
    emb_dim: int = 64
    batch_size: int = 1024
    lr: float = 0.005
    num_epochs: int = 20
    device: str = 'cuda'
    checkpoint_dir: str = 'checkpoint/'


def config_parser() -> Config:
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, choices=['movielens', 'gowalla', 'yelp'], default=Config.dataset)
    parser.add_argument('--model', type=str, choices=['lightgcn', 'mf'], default=Config.model)
    parser.add_argument('--seed', type=int, default=Config.seed)
    parser.add_argument('--min_user_cnt', type=int, default=Config.min_user_cnt)
    parser.add_argument('--min_item_cnt', type=int, default=Config.min_item_cnt)
    parser.add_argument('--split_method', type=str, choices=['random', 'leave_one_out'], default=Config.split_method)
    parser.add_argument('--test_ratio', type=float, default=Config.test_ratio)
    parser.add_argument('--val_ratio', type=float, default=Config.val_ratio)
    parser.add_argument('--num_neg_samples', type=int, default=Config.num_neg_samples)
    parser.add_argument('--loss', type=str, choices=['bpr', 'bce'], default=Config.loss)
    parser.add_argument('--num_layers', type=int, default=Config.num_layers)
    parser.add_argument('--emb_dim', type=int, default=Config.emb_dim)
    parser.add_argument('--batch_size', type=int, default=Config.batch_size)
    parser.add_argument('--lr', type=float, default=Config.lr)
    parser.add_argument('--num_epochs', type=int, default=Config.num_epochs)
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default=Config.device)
    parser.add_argument('--checkpoint_dir', type=str, default=Config.checkpoint_dir)

    args = parser.parse_args()
    config = Config(**vars(args))

    return config
