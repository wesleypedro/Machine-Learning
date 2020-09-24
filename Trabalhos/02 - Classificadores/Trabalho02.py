import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class LogisticRegressionGradientDescent():
	def __init__(self,  alpha=0.0001, epochs=200):
		self.alpha = alpha
		self.epochs = epochs

	def sigmoid(self, z):
		return 1 / (1 + np.exp(-z))

	def entropy(self, X, y, b):
		m = len(X)
		p1 = np.multiply(-y, np.log(self.sigmoid(X @ b.T)))
		p2 = np.multiply((1 - y), np.log(1 - self.sigmoid(X @ b.T)))

		somatorio = np.sum(p1 - p2)

		return somatorio/m

	def fit(self, X, y):
		n = X.shape[0]
		m = X.shape[1]

		X_ = np.c_[np.ones(n), X]
		self.beta = np.random.rand(1,m+1)

		self.custo = np.zeros(self.epochs)

		for i in range(self.epochs):
			self.beta = self.beta - (self.alpha / len(X_)) * np.sum((self.sigmoid(X_ @ self.beta.T)))
			self.custo[i] = self.entropy(X_, y, self.beta)

	def predict(self, X, limiar=0.5):
		n = X.shape[0]
		X_ = np.c_[np.ones(n), X]
		return (self.sigmoid(X_ @ self.beta.T) >= limiar).astype('int')


class NaiveBayesGaussian():
	def __init__(self):
		pass

	def fit(self):
		pass

	def predict(self):
		pass

class GaussianQuadraticDiscriminant():
	def __init__(self):
		pass

	def fit(self):
		pass

	def predict(self):
		pass

class Metrics():
	def accuracy_percentage(y_true, y_pred):
		n = len(y_true)
		c = len([i for i, j in zip(y_true, y_pred) if  i == j])
		
		return c/n

class ConfusionMatrix():
	def plot_confusion_matrix(X, y, clf):
		classes = np.unique(y)
		predicted = clf.predict(X)

		d = len(classes)

		matrix = np.zeros((d, d))

		for a, b in zip(y, predicted):
			if a == b:
				i = np.where(classes == a)
				i = i[0][0]
				matrix[i][i] += 1
			else:
				i = np.where(classes == a)
				j = np.where(classes == b)
				i = i[0][0]
				j = j[0][0]
				matrix[i][j] += 1
		
		ds_cm = pd.DataFrame(matrix, index = [i for i in classes], columns = [i for i in classes])

		plt.figure(figsize=(10,7))
		plt.title("Matriz de confusão simples")
		sns.heatmap(ds_cm)

class Dispersal():
	def plot_boundaries(X, y, clf):
		# Plot the decision boundary. For that, we will assign a color to each
		# point in the mesh [x_min, x_max]x[y_min, y_max].
		x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
		y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
		h = .02  # step size in the mesh
		xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
		Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

		# Put the result into a color plot
		Z = Z.reshape(xx.shape)
		plt.figure(1, figsize=(4, 3))
		plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

		# Plot also the training points
		plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
		plt.xlabel('X0')
		plt.ylabel('X1')

		plt.xlim(xx.min(), xx.max())
		plt.ylim(yy.min(), yy.max())
		plt.xticks(())
		plt.yticks(())

		plt.show()