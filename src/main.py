from data.process import DataProcessor
from utils import prepare_training, build_model_trainer


def main():
    config = prepare_training()
    processor = DataProcessor(config)
    data = processor.process()
    trainer = build_model_trainer(config, data)
    trainer.train(data)
    trainer.test(data)


if __name__=='__main__':
    main()