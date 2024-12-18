import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree

class AdaSoftDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self, alpha=0.1, n_runs=100):
        super().__init__()
        self.alpha = alpha  # Base alpha used when information gain is 0
        self.n_runs = n_runs

    def _calculate_information_gain(self, tree, node_id):
        """Calculate the information gain at a specific node."""
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        # If it's a leaf node, no split, no information gain
        if left_child == _tree.TREE_UNDEFINED or right_child == _tree.TREE_UNDEFINED:
            return 0.0

        # Impurity (Gini) values before and after the split
        impurity_before = tree.impurity[node_id]
        impurity_left = tree.impurity[left_child]
        impurity_right = tree.impurity[right_child]

        # The weighted average impurity after the split
        n_left = tree.n_node_samples[left_child]
        n_right = tree.n_node_samples[right_child]
        total_samples = tree.n_node_samples[node_id]
        
        impurity_after = (n_left / total_samples) * impurity_left + (n_right / total_samples) * impurity_right

        # Information gain is the difference in impurity
        return impurity_before - impurity_after

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

                # Calculate information gain for current node
                info_gain = self._calculate_information_gain(tree, node_id)
                # Adjust alpha based on information gain: more certainty = less randomness
                dec_alpha = self.alpha * (1 - info_gain)

                left_direction = sample[feature] <= threshold
                if np.random.rand() <dec_alpha:
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
