import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree

class SoftDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self, alpha=0.0001, n_runs=10):
        super().__init__()
        self.alpha = alpha
        self.n_runs = n_runs

    def _soft_predict(self, tree, X):
        """
        Helper function that implements the "soft split" logic for one prediction.
        """
        n_samples = X.shape[0]
        n_classes = self.n_classes_
        # Array to hold the probabilities of each sample for each class
        prob = np.zeros((n_samples, n_classes))
        
        for sample_idx in range(n_samples):
            sample = X[sample_idx]
            node_id = 0  # start at the root node
            # Traverse the tree
            while tree.feature[node_id] != _tree.TREE_UNDEFINED:
                feature = tree.feature[node_id]
                threshold = tree.threshold[node_id]
                # Get the split decision (soft split)
                left_direction = sample[feature] <= threshold
                # Soft routing: with probability 1-alpha go the normal direction, with alpha go the opposite direction
                if np.random.rand() < self.alpha:
                    left_direction = not left_direction
                
                # Move to the left or right child node
                if left_direction:
                    node_id = tree.children_left[node_id]
                else:
                    node_id = tree.children_right[node_id]

            # Now we're at a leaf node, the probability of each class is stored there
            prob[sample_idx] = tree.value[node_id].flatten() / tree.value[node_id].sum()

        return prob

    def predict_proba(self, X):
        """
        Override the predict_proba function to implement soft splits.
        """
        # Get the decision tree structure
        tree = self.tree_
        # Run the soft prediction n_runs times
        all_probs = np.zeros((X.shape[0], self.n_classes_))

        for _ in range(self.n_runs):
            probs = self._soft_predict(tree, X)
            all_probs += probs
        
        # Average the probabilities over n_runs
        return all_probs / self.n_runs