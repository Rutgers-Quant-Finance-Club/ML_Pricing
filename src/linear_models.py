import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

class OLS:
    def __init__(self):
        self.weights = None


    def fit(self, X, Y):
        # find weights in the form of a 1D numpy array such that, for each stock, multiply them by feature_count weights 
        # where each column has its own weight. Thus, there will be a feature_count amount of weights - and a row_number 
        # amount of multiplications for each weight. The sum of all of these weights and values in the columns
        # is being best adjusted to equal the return. These weights are the closest possible fit - a minimization of the sum of
        # squared errors between X @ w and Y. This is OLS.
        self.weights = np.linalg.inv(X.T @ X) @ X.T @ Y

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model not fitted yet. Call fit(input: X, target: Y) first.")
        # multiply input df X by the weights such that Xw = Y where Y represents predictions
        predictions = X @ self.weights
        return predictions
    

class ElasticNetModel:
    def __init__(self, alpha=0.001, l1_ratio=0.5):
        # note that elastic net is very similar to OLS, just with penalization - alpha signifies how strong penalization is, 
        # l1_ratio is the different types alpha "acts through", that is, if l1_ratio = 1, it's all l1 - and this zeroes out
        # large features, or l2, which shrinks them, which is represented by l1_ratio = 0. These are the hyperparameters,
        # and this is the elastic net.
        self.weights = None
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.model = None
    
    # buffers the model using the ElasticNet class as provided by scikit-learn. 
    def fit(self, X, Y):
        self.model = ElasticNet(self.alpha, self.l1_ratio)
        self.model.fit(X, Y)
    
    def predict(self, X):
        # provides a predictions numpy array, given any data frame X such that X represents features/input to the model
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit(input: X, target: Y) first.")
        predictions = self.model.predict(X)
        return predictions
    

    
    