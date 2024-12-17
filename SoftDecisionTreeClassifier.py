import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree

class SoftDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self, alpha=0.0001, n_runs=10):
        super().__init__()
        self.alpha = alpha
        self.n_runs = n_runs

    def _soft_predict(self, tree, X):
        n_samples = X.shape[0]
        n_classes = self.n_classes_
        prob = np.zeros((n_samples, n_classes))
        
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

        return prob

    def predict_proba(self, X):
        tree = self.tree_

        all_probs = np.zeros((X.shape[0], self.n_classes_))

        for _ in range(self.n_runs):
            probs = self._soft_predict(tree, X)
            all_probs += probs
        
        return all_probs / self.n_runs