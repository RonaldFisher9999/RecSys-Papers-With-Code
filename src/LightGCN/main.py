from LightGCN.args import parser
from LightGCN.model import LightGCN
from LightGCN.process import DataProcessor

def main():
    config = parser.parse_args()
    for k, v in vars(config).items():
        print(k, v)
    processor = DataProcessor(
        config.dataset,
        config.min_user_cnt,
        config.min_item_cnt,
        config.test_ratio,
        config.valid_ratio      
    )
    edge_index, train, val, test, num_users, num_items = processor.process()
    # print(edge_index)
    # print(edge_index.shape)
    # print(train)
    # print(val)
    # print(test)
    # print(n_users)
    # print(n_items)
    
    model = LightGCN(num_users, num_items, config.num_layers, config.emb_dim).to(config.device)
    print(model)
    
if __name__=='__main__':
    main()