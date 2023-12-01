from args import parser
from model import LightGCN
from process import DataProcessor

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
    edge_index, train, val, test, n_users, n_items = processor.process()
    print(edge_index)
    print(edge_index.shape)
    print(train)
    print(val)
    print(test)
    print(n_users)
    print(n_items)
    
if __name__=='__main__':
    main()