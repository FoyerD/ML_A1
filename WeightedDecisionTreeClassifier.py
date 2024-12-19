import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree

class WeightedSoftDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self, alpha=0.0001, n_runs=10):
        super().__init__()
        self.alpha = alpha
        self.n_runs = n_runs

    def _get_leaf_weight(self, tree, node_id):
        impurity = tree.impurity[node_id]
        # lower impurity -> higher weight
        # normalize_weight = 0.0001
        # return 1 / (impurity + normalize_weight)  # Adding a small value to avoid division by zero
        return 1 - impurity

    def _soft_predict(self, tree, X):
        n_samples = X.shape[0]
        n_classes = self.n_classes_
        prob = np.zeros((n_samples, n_classes))
        weights = np.zeros(n_samples)

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
  
            prob[i] = tree.value[node_id].flatten() / tree.value[node_id].sum()
            weights[i] = self._get_leaf_weight(tree, node_id)
            prob[i] *= weights[i]

        return prob, weights

    def predict_proba(self, X):
        tree = self.tree_
        n_samples = X.shape[0]
        all_probs = np.zeros((n_samples, self.n_classes_))
        sum_weights = np.zeros(n_samples)

        for _ in range(self.n_runs):
            probs, weights = self._soft_predict(tree, X)
            all_probs += probs
            sum_weights += weights
        
        for i in range(n_samples):
            all_probs[i] = all_probs[i] / sum_weights[i]

        return all_probs

