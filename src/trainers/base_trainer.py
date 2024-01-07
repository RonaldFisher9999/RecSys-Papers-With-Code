class BaseModelTrainer:
    def train(self):
        raise NotImplementedError
    
    def test(self):
        raise NotImplementedError
    
    def _build_model(self):
        raise NotImplementedError
    
    def _fit(self):
        raise NotImplementedError
    
    def _validate(self):
        raise NotImplementedError
    
    def _update_best_model(self):
        raise NotImplementedError
    
    def _load_best_model(self):
        raise NotImplementedError