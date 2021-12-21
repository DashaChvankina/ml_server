import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor
import timeit


class RandomForestMSE:
    def __init__(
        self, n_estimators, max_depth=None, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.param = trees_parameters
        self.models = []
        self.features = []

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        X_val : numpy ndarray
            Array of size n_val_objects, n_features
        y_val : numpy ndarray
            Array of size n_val_objects
        """
        if self.feature_subsample_size is None:
            self.feature_subsample_size = X.shape[1] // 3
        if not(X_val is None):
            history = {'time':[], 'rmse': []}
            start_time = timeit.default_timer()
            y_pred = np.zeros(y_val.shape)
        for i in range(self.n_estimators):
            Ind = np.random.randint(low=0, high=X.shape[0], size=X.shape[0])
            Feat = np.random.choice(X.shape[1], size=self.feature_subsample_size, replace=False)
            model = DecisionTreeRegressor(max_depth=self.max_depth, **self.param)
            model.fit(np.take(X[Ind], Feat, axis=1), y[Ind])
            self.models.append(model)
            self.features.append(Feat)
            if not(X_val is None):
                y_pred += model.predict(np.take(X_val, Feat, axis=1))
                rmse = np.sqrt(((((y_pred/(i+1)) - y_val)**2).mean()))
#                 y_pred = self.predict(X_val)
#                 rmse = math.sqrt((((y_pred - y_val)**2).mean()))
                history['rmse'].append(rmse)
                history['time'].append(timeit.default_timer() - start_time)
        if not(X_val is None):
            return history

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        pred = np.empty((X.shape[0], self.n_estimators))
        for i in range(len(self.models)):
            pred[:, i] = self.models[i].predict(np.take(X, self.features[i], axis=1))
        return pred.mean(axis=1)

class GradientBoostingMSE:
    def __init__(
        self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        learning_rate : float
            Use alpha * learning_rate instead of alpha
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.param = trees_parameters
        self.models = []
        self.alpha = []
        self.features = []
        
    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        """
        
        if self.feature_subsample_size is None:
            self.feature_subsample_size = X.shape[1] // 3
        if not(X_val is None):
            history = {'time':[], 'rmse': []}
            start_time = timeit.default_timer()
        Feat = np.random.choice(X.shape[1], size=self.feature_subsample_size, replace=False)
        g = y
        model = DecisionTreeRegressor(max_depth=self.max_depth, **self.param)
        model.fit(np.take(X, Feat, axis=1), g)
        self.models.append(model)
        self.alpha.append(1)
        self.features.append(Feat)
        pred = self.predict(X)
        g = y - pred
        if not(X_val is None):
                y_pred = model.predict(X_val[:, Feat]) 
                rmse = np.sqrt((((y_pred - y_val) **2).mean()))
                history['rmse'].append(rmse)
                history['time'].append(timeit.default_timer() - start_time)
        for i in range(self.n_estimators - 1):
            Feat = np.random.choice(X.shape[1], size=self.feature_subsample_size, replace=False)
            model = DecisionTreeRegressor(max_depth=self.max_depth, **self.param)
            model.fit(np.take(X, Feat, axis=1), g)
            y_new = model.predict(np.take(X, Feat, axis=1))
            def func(alpha):
                return ((pred + alpha * y_new - y) ** 2).sum()
#             alpha = minimize_scalar(lambda x: ((pred + x * self.learning_rate * y_new - y) ** 2).mean()).x
            alpha = minimize_scalar(func).x
            self.models.append(model)
            self.alpha.append(alpha)
            self.features.append(Feat)
            pred += self.learning_rate * alpha * y_new
            g = y - pred
            if not(X_val is None):
                y_pred += alpha * self.learning_rate * model.predict(np.take(X_val, Feat, axis=1))
                rmse = np.sqrt((((y_pred - y_val) **2).mean()))
                history['rmse'].append(rmse)
                history['time'].append(timeit.default_timer() - start_time)
        if not(X_val is None):
            return history

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        pred = self.alpha[0] * self.models[0].predict(np.take(X, self.features[0], axis=1))
        for i in range(1, len(self.models)):
            pred += self.alpha[i] * self.learning_rate * self.models[i].predict(np.take(X, self.features[i], axis=1))
        return pred