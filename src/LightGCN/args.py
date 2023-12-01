import argparse


parser = argparse.ArgumentParser(description='Config for LightGCN')

parser.add_argument('--dataset', type=str, default='movielens', choices=['movielens'])
parser.add_argument('--min_user_cnt', type=int, default=5)
parser.add_argument('--min_item_cnt', type=int, default=5)
parser.add_argument('--test_ratio', type=float, default=0.2)
parser.add_argument('--valid_ratio', type=float, default=0.1)
parser.add_argument('--num_neg_samples', type=int, default=1)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--emb_dim', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--lambda_reg', type=float, default=0.0005)
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])