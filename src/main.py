from config_parser import parser
from data.process import LightGCNDataProcessor
from models.lightgcn import LightGCN, LightGCNTrainer
from utils import set_seeds


class RecSysTrainer:
    def __init__(self):
        self.config = parser.parse_args()
        
    def train(self):
        self._prepare()
        edge_index, train, valid, test, n_users, n_items = self._process_data()
        model = self._build_model(n_users, n_items)
        model_trainer = self._build_model_trainer(n_users, n_items)
        model = model_trainer.train(model, edge_index, train, valid)
        self._inference(model, test)
    
    def _prepare(self):
        for k, v in vars(self.config).items():
            print(k, v)
        set_seeds(self.config.seed)
    
    def _process_data(self):
        processor = LightGCNDataProcessor(
            self.config.dataset,
            self.config.min_user_cnt,
            self.config.min_item_cnt,
            self.config.test_ratio,
            self.config.valid_ratio,
            )
        edge_index, train, valid, test, n_users, n_items = processor.process()
        
        print(f'Number Of Users: {n_users}')
        print(f'Number Of Items: {n_items}')
    
        return edge_index, train, valid, test, n_users, n_items
    
    def _build_model(self, n_users, n_items):
        return LightGCN(n_users, n_items, self.config.num_layers, self.config.emb_dim).to(self.config.device)
        
    def _build_model_trainer(self, n_users, n_items):
        return LightGCNTrainer(self.config.num_epochs, n_users, n_items, self.config.num_neg_samples,
                              self.config.lr, self.config.batch_size, self.config.lambda_reg,
                              self.config.device, self.config.checkpoint_dir, self.config.dataset)
    
    def _inference(self, model, test):
        pass
    
    
if __name__=='__main__':
    trainer = RecSysTrainer()
    trainer.train()