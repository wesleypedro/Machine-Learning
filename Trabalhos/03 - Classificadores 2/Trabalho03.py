import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline


class MLPClassifier(object):
    def __init__(self, alpha=.0001, hiden_size=2, epoch=10):
        self._alpha = alpha
        self._hiden_size = hiden_size
        self._epoch = epoch
        self._output_size = 1
        self._input_size = None

    def _sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def _d_sigmoid(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def _predict_value(self, x):
        return 1 if x >= 0.5 else 0
    
    def _normalize(self, x):
        x = np.asarray(x)
        return (x - x.min()) / (np.ptp(x))

    def fit(self, X, y):
        # Normalização dos dados para que possa convergir mais rápido
        X = self._normalize(X)
        
        n = X.shape[0]

        bias = (-1) * (np.ones(n))
        X = np.c_[bias, X]
        
        self._input_size = X.shape[1]

        self.w = np.random.rand(self._hiden_size, self._input_size)
        self.m = np.random.rand(self._output_size, self._hiden_size+1)
        
        # Inicializando aqui para poupar processamento repetitivo mais abaixo
        deltai = [[0. for i in range(self._hiden_size)]] * n
        
        for _ in range(self._epoch):
            # Forward Propagation
            ui = self.w.dot(X.T)
            
            zi = self._sigmoid(ui)
            zi = np.array(zi).T
            zi = np.c_[(-1)*(np.ones(n)), zi]
        
            uk = self.m.dot(zi.T)

            yk = self._sigmoid(uk)
            
            # Backpropagation
            ek = y - yk

            duk = self._d_sigmoid(uk)
            duk = np.array(duk)
            
            deltak = ek * duk
        
            for j in range(n):
                for i in range(self._hiden_size):
                    sum_ = 0
                    for k in range(self._output_size):
                        sum_ += duk[k,j] * self.m[k,i]
                    deltai[j][i] = self._d_sigmoid(ui[i,j]) * sum_
            deltai = np.array(deltai)

            self.m += self._alpha * (deltak.dot(zi))
            self.w += self._alpha * (deltai.T.dot(X))


    def predict(self, X):
        # Normalização dos dados para que possa convergir mais rápido
        X = self._normalize(X)
        
        n = X.shape[0]

        bias = (-1) * (np.ones(n))
        X = np.c_[bias, X]

        ui = self.w.dot(X.T)
            
        zi = self._sigmoid(ui)
        zi = np.array(zi).T
        zi = np.c_[(-1)*(np.ones(n)), zi]

        uk = self.m.dot(zi.T)

        yk = self._sigmoid(uk)
        
        yk = [self._predict_value(e) for e in yk[0]]
        return np.array(yk)


class KNN(object):
    def __init__(self, n_neighbors=5, metric='euclidian'):
        self._n_neighbors = n_neighbors
        self._metric = metric
        self._data = None

    def _euclidian(self, r1, r2):
        distance = 0.
        for i in range(len(r1)-1):
            distance += (r1[i] - r2[i])**2
        return np.sqrt(distance)
    
    def _manhattan(self, r1, r2):
        distance = 0.
        for i in range(len(r1)-1):
            distance += abs(r1[i]-r2[i])
        return distance
    
    def _get_neighbors(self, test_row):
        distances = list()
        distance = 0
        for row in self._data:
            if self._metric == 'euclidian':
                distance = self._euclidian(test_row, row)
            else:
                distance = self._manhattan(test_row, row)
            distances.append((row, distance))
        distances.sort(key=lambda tup:tup[1])
        neighbors = list()
        for i in range(self._n_neighbors):
            neighbors.append(distances[i][0])
        return neighbors
    
    def fit(self, X, y):
        self._data = np.c_[X,y]
    
    def predict(self, X):
        y_pred = list()
        for xi in X:
            neighbors = self._get_neighbors(xi)
            output_values = [row[-1] for row in neighbors]
            # Semelhante a ir pelas médias, pois o que tiver em maior quantidade é o que vai 'arrastar' para o valor final
            prediction = max(set(output_values), key=output_values.count)
            y_pred.append(prediction)
        return y_pred

    
class Validation(object):
    def __init__(self):
        pass
    
    def kFold(self, X, y, k, metodo):
        metrics = Metrics()
        n = X.shape[0]
        
        subset_size = round(n/k)
        
        X_subsets = [X[e:e+subset_size,:] for e in range(0, n, subset_size)]
        y_subsets = [y[e:e+subset_size] for e in range(0, n, subset_size)]
        
        errors = []
        
        for i in range(k):
            test_X = X_subsets[i]
            test_y = y_subsets[i]
            
            train_X = []
            for j in range(k):
                if i != j:
                    for e in X_subsets[j]:
                        train_X.append(e)
            
            train_y = []
            for j in range(k):
                if i != j:
                    for e in y_subsets[j]:
                        train_y.append(e)
    
            train_X = np.array(train_X)
            train_y = np.array(train_y)
            
            metodo.fit(train_X, train_y)
            predicted = metodo.predict(test_X)
            
            errors.append(metrics.accuracy_percentage(test_y, predicted))
        
        return errors


class Metrics(object):
	def accuracy_percentage(self, y_true, y_pred):
		n = len(y_true)
		c = len([i for i, j in zip(y_true, y_pred) if  i == j])
		
		return c/n


class ConfusionMatrix(object):
	def plot_confusion_matrix(self, X, y, clf):
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


class Dispersal(object):
	def plot_boundaries(self, X, y, clf):
		# Plot the decision boundary. For that, we will assign a color to each
		# point in the mesh [x_min, x_max]x[y_min, y_max].
		x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
		y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
		h = .02  # step size in the mesh
		xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
		Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

		# Put the result into a color plot
		Z = Z.reshape(xx.shape)
		plt.figure(1, figsize=(6, 5))
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
