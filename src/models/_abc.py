from abc import ABC

class SkLearnModelsABC(ABC):
    def __init__(self, ):
        self.model = ...
    
    def fit(self, X, y):
        self.model.fit(X, y)  # For sklearn
        
    def predict(self, X):
        return self.model.predict(X) # For sklearn