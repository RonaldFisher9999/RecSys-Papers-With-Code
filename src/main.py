from config import config_parser
from data.process import process_data
from utils import prepare_training, build_model_trainer
from eval import eval_model


def main():
    config = config_parser()
    prepare_training(config)
    data, data_info = process_data(config)
    trainer = build_model_trainer(config, data, data_info)
    model = trainer.train(data)
    exit()
    eval_model(model, data)
    

if __name__=='__main__':
    main()