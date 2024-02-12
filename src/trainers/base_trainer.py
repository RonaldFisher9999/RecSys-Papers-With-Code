from abc import ABC, abstractmethod


class BaseModelTrainer(ABC):
    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def _build_model(self):
        raise NotImplementedError

    @abstractmethod
    def _build_loader(self):
        raise NotImplementedError

    @abstractmethod
    def _fit(self):
        raise NotImplementedError

    @abstractmethod
    def _validate(self):
        raise NotImplementedError

    @abstractmethod
    def _update_best_model(self):
        raise NotImplementedError

    @abstractmethod
    def _load_best_model(self):
        raise NotImplementedError
