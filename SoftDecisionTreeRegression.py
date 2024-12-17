import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import _tree

class SoftDecisionTreeRegressor(DecisionTreeRegressor):
    def __init__(self, alpha=0.001, n_runs=1):
        super().__init__()
        self.alpha = alpha
        self.n_runs = n_runs

    def _soft_predict(self, tree, X):
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        
        for i in range(n_samples):
            sample = X[i]
            node_id = 0  

            while tree.feature[node_id] != _tree.TREE_UNDEFINED:
                feature = tree.feature[node_id]
                threshold = tree.threshold[node_id]

                left_direction = sample[feature] <= threshold
                if np.random.rand() < self.alpha:
                    left_direction = not left_direction
                
                if left_direction:
                    node_id = tree.children_left[node_id]
                else:
                    node_id = tree.children_right[node_id]

            predictions[i] = tree.value[node_id].flatten()

        return predictions

    def predict(self, X):
        tree = self.tree_
        all_predictions = np.zeros(X.shape[0])

        for _ in range(self.n_runs):
            preds = self._soft_predict(tree, X)
            all_predictions += preds
        
        return all_predictions / self.n_runs
