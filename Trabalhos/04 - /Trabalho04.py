import numpy as np


class KMeans(object):
    def __init__(self, k=5, ite=100):
        self._k = k
        self._ite = ite
        self.inertia = -1

    def _euclidian(self, a, b):
        return (((a-b)**2).sum()**(0.5))

    def _initial_clusters(self, X, k):
        m = X.shape[1]

        _max = np.max(X, axis=0)
        _min = np.min(X, axis=0)

        centroids =  np.random.uniform(_min, _max, (k, m))

        return centroids

    def _get_nearest_centroid(self, x, centroids):
        return np.argmin(list(map(lambda c : self._euclidian(x, c), centroids)))

    def _compute_nearest(self, X, centroids):
        index = list()

        for x in X:
            index.append(self._get_nearest_centroid(x, centroids))

        return np.array(index)

    def _get_inertia(self, X, clusters_centers, labels):
        dist = list()
        abs_dist = list()

        for i, data in enumerate(X):
            euclidian_dist = self._euclidian(data, clusters_centers[labels[i]])
            dist.append(euclidian_dist**2)
            abs_dist.append(euclidian_dist)

        return sum(dist), abs_dist

    def _update_centroids(self, X, centroids, labels):
        new_centroids = list()
        points = list()

        for i in range(len(centroids)):
            points = list()
            for j, data in enumerate(X):
                if labels[j] == i:
                    points.append(data)
            points_ = np.array(points)
            if len(points_) > 0:
                new_centroids.append(np.mean(points_, axis=0))
            else:
                new_centroids.append(centroids[i])

        return np.array(new_centroids)
    
    def _get_mean_distance(self, distances):
        distances = np.array(distances)
        return [np.mean(col) for col in distances.T]

    def fit(self, X):
        _distance = list()
        
        self.clusters_centers = self._initial_clusters(X, self._k)

#         self.labels = self._get_nearest_centroid(X, self.clusters_centers)
        self.labels = self._compute_nearest(X, self.clusters_centers)

        old_inertia, abs_dist = self._get_inertia(X, self.clusters_centers, self.labels)
        _distance.append(abs_dist)
        
        self.mean_distance = list()

        for _ in range(self._ite):
            self.clusters_centers = self._update_centroids(X, self.clusters_centers, self.labels)
            self.labels = self._compute_nearest(X, self.clusters_centers)
            self.inertia, abs_dist = self._get_inertia(X, self.clusters_centers, self.labels)
            
            _distance.append(abs_dist)
            
            if self.inertia == old_inertia:
                self.mean_distance = self._get_mean_distance(_distance)
                break
            old_inertia = self.inertia
        
        self.mean_distance = self._get_mean_distance(_distance)

    def predict(self, X):
        return self._compute_nearest(X, self.clusters_centers)


class Node:
    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None
    

class DecisionTree(object):
    def __init__(self, max_depth):
        self._max_depth = max_depth
        self._depth = 0

    def _classes(self, y):
        return sum(y.unique())
    
    def _gini(self, labels):
        values, counts = np.unique(labels, return_counts=True)
        sum_ = 0.
        total_sum = np.sum(counts)
        for c in counts:
            sum_ += (c/total_sum)**2
        return 1 - sum_
    
    def _best_split(self, X, y):
        m = y.size
        if m <= 1:
            return None, None
        
        num_parent = [np.sum(y == c) for c in range(self.n_classes)]
        
        best_gini = self._gini(y)
#         best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None
        
        for idx in range(self.n_features):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            
            num_left = [0] * self.n_classes
            num_right = num_parent.copy()
            for i in range(1, m):
                c = int(classes[i - 1])
                num_left[c-1] += 1
                num_right[c-1] -= 1
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.n_classes))
                gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(self.n_classes))

                gini = (i * gini_left + (m - i) * gini_right) / m

                if thresholds[i] == thresholds[i - 1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2

        return best_idx, best_thr
    
    def _build_tree(self, X, y, depth=0):
        number_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]
        predicted_class = np.argmax(number_samples_per_class)
        
        node = Node(
            gini=self._gini(y),
            num_samples=y.size,
            num_samples_per_class=number_samples_per_class,
            predicted_class=predicted_class
        )
        
        if depth < self._max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._build_tree(X_left, y_left, depth + 1)
                node.right = self._build_tree(X_right, y_right, depth + 1)
                
        return node

    def _predict(self, X):
        node = self.tree
        
        while node.left:
            if X[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class
    
    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.n_classes = len(set(y))
        self.tree = self._build_tree(X, y)
    
    def predict(self, X):
        return [self._predict(inputs) for inputs in X]


class PCA(object):
    def __init__(self, dim):
        self._dim = dim

    def _standardization(self, X):
        mi = np.mean(X, axis=0)
        sigma = np.sqrt(np.mean((X-mi)**2))
        z = (X-mi)/sigma

        return z

    def _covariance_matrix(self, X):
#         covariance = np.cov(X)
        n = X.shape[0]
        X_mean = np.mean(X, axis=0)
        covariance = np.dot((X - X_mean).T, (X - X_mean)) / (n - 1)

        return covariance

    def fit(self, X):
        X = self._standardization(X)
        covariance = self._covariance_matrix(X)
        eigenvalues, eigenvectors = np.linalg.eig(covariance)

        index = (-eigenvalues).argsort()[:self._dim]
        self.W = eigenvectors[:, index]
        self.variance = list()
        self.variance = np.array(self.variance)
        for i in index:
            self.variance = np.append(self.variance, eigenvalues[i] / eigenvalues.sum())

    def transform(self, X):
        X = self._standardization(X)
        return (np.dot(self.W.T, X.T)).T

    
class Metrics(object):
	def accuracy_percentage(self, y_true, y_pred):
		n = len(y_true)
		c = len([i for i, j in zip(y_true, y_pred) if  i == j])
		
		return c/n

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