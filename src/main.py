from args import parser
from LightGCN.model import LightGCN
from LightGCN.process import DataProcessor
from LightGCN.trainer import LightGCNTrainer
from utils import set_seeds

import numpy as np


    
def main():
    config = parser.parse_args()
    for k, v in vars(config).items():
        print(k, v)
    set_seeds(config.seed)
    processor = DataProcessor(
        config.dataset,
        config.min_user_cnt,
        config.min_item_cnt,
        config.test_ratio,
        config.valid_ratio      
    )
    edge_index, train, valid, test, num_users, num_items = processor.process()
    # print(edge_index)
    # print(edge_index.shape)
    # print(train)
    # print(valid)
    # print(test)
    # print(n_users)
    # print(n_items)
    
    model = LightGCN(num_users, num_items, config.num_layers, config.emb_dim).to(config.device)
    print(model)
    
    trainer = LightGCNTrainer(config.num_epochs, num_users, num_items, config.num_neg_samples,
                              config.lr, config.batch_size, config.lambda_reg, config.device, config.checkpoint_dir)
    trainer.train(model, edge_index, train, valid)
    
if __name__=='__main__':
    main()