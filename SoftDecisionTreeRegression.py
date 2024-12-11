import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import _tree

class SoftDecisionTreeRegressor(DecisionTreeRegressor):
    def __init__(self, alpha=0.001, n_runs=100):
        super().__init__()
        self.alpha = alpha
        self.n_runs = n_runs

    def _soft_predict(self, tree, X):
        """
        Helper function that implements the "soft split" logic for one prediction.
        """
        n_samples = X.shape[0]
        # Array to hold the predictions for each sample
        predictions = np.zeros(n_samples)
        
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

            # Now we're at a leaf node, the prediction value is stored there
            predictions[sample_idx] = tree.value[node_id].flatten()

        return predictions

    def predict(self, X):
        """
        Override the predict function to implement soft splits for regression.
        """
        # Get the decision tree structure
        tree = self.tree_
        # Run the soft prediction n_runs times
        all_predictions = np.zeros(X.shape[0])

        for _ in range(self.n_runs):
            preds = self._soft_predict(tree, X)
            all_predictions += preds
        
        # Average the predictions over n_runs
        return all_predictions / self.n_runs
