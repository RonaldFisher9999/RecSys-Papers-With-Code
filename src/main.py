from args import parser
from models.lightgcn import LightGCNDataProcessor, LightGCN, LightGCNTrainer
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
    edge_index, train, valid, test, num_users, num_items = processor.process()
    
    model = LightGCN(num_users, num_items, config.num_layers, config.emb_dim).to(config.device)
    print(model)
    
    trainer = LightGCNTrainer(config.num_epochs, num_users, num_items, config.num_neg_samples,
                              config.lr, config.batch_size, config.lambda_reg, config.device, config.checkpoint_dir)
    trainer.train(model, edge_index, train, valid)
    
if __name__=='__main__':
    main()