import numpy as np

class univariate_linear_regression_analytical_method():
    def __init__(self):
        self.beta0 = None
        self.beta1 = None
    
    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        
        x_mean = X.mean()
        y_mean = y.mean()
        
        x_dif = X-x_mean
        y_dif = y-y_mean
        
        num = sum(x_dif*y_dif)
        den = sum(x_dif*x_dif)
        
        self.beta1 = num/den
        self.beta0 = y_mean-(self.beta1*x_mean)     
    
    def predict(self, X):
        return self.beta0+(self.beta1*X)

class univariate_linear_regression_descending_gradient():
    def __init__(self, rate=0.001, epochs=10):
        self.rate = rate
        self.epochs = epochs

    def fit(self, X, y):
        pass

    def predict(self, X):
        return 

class multivariate_linear_regression_analytical_method():
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

class multivariate_linear_regression_descending_gradient():
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

class multivariate_linear_regression_stochastic_descending_gradient():
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

class quadratic_regression_using_multiple_regression():
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

class cubic_regression_using_multiple_regression():
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

class multivariate_regularized_linear_regression_descending_gradient():
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass


class functions():
    def __init__(self):
        pass
    
    def mse(self, y_true, y_predict):
        e = y_true-y_predict
        e2 = e*e
        return (sum(e2)).mean()

    def rss(self, y_true, y_predict):
        return np.sum((y_true-y_predict)**2)
    
    def tss(self, y_true, y_predict):
        y_mean = np.mean(y_predict)
        return np.sum((y_true-y_mean)**2)
    
    def r2(self, y_true, y_predict):
        return 1-(self.rss(y_true, y_predict)/self.tss(y_true, y_predict))
