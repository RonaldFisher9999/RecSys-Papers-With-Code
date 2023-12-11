from args import parser
from data.process import LightGCNDataProcessor
from models.lightgcn import LightGCN, LightGCNTrainer
from utils import set_seeds

 
def main():
    config = parser.parse_args()
    for k, v in vars(config).items():
        print(k, v)
    set_seeds(config.seed)
    processor = LightGCNDataProcessor(
        config.dataset,
        config.min_user_cnt,
        config.min_item_cnt,
        config.test_ratio,
        config.valid_ratio      
    )
    edge_index, train, valid, test, n_users, n_items = processor.process()
    print(f'Number Of Users: {n_users}')
    print(f'Number Of Items: {n_items}')
    
    model = LightGCN(n_users, n_items, config.num_layers, config.emb_dim).to(config.device)
    trainer = LightGCNTrainer(config.num_epochs, n_users, n_items, config.num_neg_samples,
                              config.lr, config.batch_size, config.lambda_reg,
                              config.device, config.checkpoint_dir, config.dataset)
    trainer.train(model, edge_index, train, valid)
    
if __name__=='__main__':
    main()