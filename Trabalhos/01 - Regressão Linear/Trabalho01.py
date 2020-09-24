import numpy as np


# 1.a
class univariate_linear_regression_analytical_method():
    def __init__(self):
        self.beta0 = None
        self.beta1 = None

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)

        x_mean = X.mean()
        y_mean = y.mean()

        x_dif = X - x_mean
        y_dif = y - y_mean

        num = sum(x_dif * y_dif)
        den = sum(x_dif * x_dif)

        self.beta1 = num / den
        self.beta0 = y_mean - (self.beta1 * x_mean)

    def predict(self, X):
        return self.beta0 + (self.beta1 * X)


# 1.b
class univariate_linear_regression_descending_gradient():
    def __init__(self, alpha=0.001, epochs=10):
        self.alpha = alpha
        self.epochs = epochs
        self.w0 = 1
        self.w1 = 1

    def fit(self, X, y):
        n = X.shape

        for _ in range(self.epochs):
            yh = self.w1 * X + self.w0

            e = y - yh

            self.w0 = self.w0 + self.alpha * (sum(e) / n)
            self.w1 = self.w1 + self.alpha * (sum(e * X) / n)

    def predict(self, X):
        return self.w1 * X + self.w0


# 1.c
class multivariate_linear_regression_analytical_method():
    def __init__(self):
        self.beta = None
        self.ones = None

    def fit(self, X, y):
        self.ones = np.ones(X.shape[0])

        self.X = np.column_stack(self.ones, X)
        XTXi = np.linalg.inv(self.X.T.dot(self.X))

        self.beta = XTXi.dot(self.X.T).dot(y)

    def predict(self, X):
        X = np.column_stack(self.ones, X)
        return X.dot(self.beta)


# 1.d
class multivariate_linear_regression_descending_gradient():
    def __init__(self, alpha=0.001, epochs=10):
        self.alpha = alpha
        self.epochs = epochs
        self.w = None
        self.ones = None
        self.w_pasts = []

    def fit(self, X, y):
        self.w = np.zeros(X.shape[0])

        self.ones = np.zeros(X.shape[0])
        self.X = np.column_stack(self.ones, X)

        for _ in range(self.ephocs):
            e = y - self.w.t.dot(self.X)
            self.w = self.w + self.alpha * (sum(e * self.X) / len(e))
            self.w_pasts.append(self.w)

    def predict(self, X):
        pass

# 1.e
class multivariate_linear_regression_stochastic_descending_gradient():
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass


# 1.f
class quadratic_regression_using_multiple_regression():
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass


# 1.g
class cubic_regression_using_multiple_regression():
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass


# 1.h
class multivariate_regularized_linear_regression_descending_gradient():
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass


# 2
class functions():
    def __init__(self):
        pass
    
    # 2.a
    def mse(self, y_true, y_predict):
        e = y_true-y_predict
        e2 = e*e
        return (sum(e2)).mean()

    def rss(self, y_true, y_predict):
        return np.sum((y_true-y_predict)**2)
    
    def tss(self, y_true, y_predict):
        y_mean = np.mean(y_predict)
        return np.sum((y_true-y_mean)**2)
    
    # 2.b
    def r2(self, y_true, y_predict):
        return 1-(self.rss(y_true, y_predict)/self.tss(y_true, y_predict))
