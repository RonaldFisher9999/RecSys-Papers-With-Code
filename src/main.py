from data.process import process_data
from utils import prepare_training, build_model_trainer
from eval import eval_model


def main():
    config = prepare_training()
    data = process_data(config)
    trainer = build_model_trainer(config, data)
    trainer.train(data)
    trainer.test(data)


if __name__=='__main__':
    main()