import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import _tree
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
from ucimlrepo import fetch_ucirepo
  
# ---------- Implimintation ----------

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


# ---------- General Data Functionallity ----------

def test_model_UCI(dataset, transform_label_func, encode=False):
    X = dataset.data.features
    if encode:
        X = pd.get_dummies(X, drop_first=False)
    y = dataset.data.targets
    test_model_X_y(X, y, transform_label_func)

def test_model_csv(df, transform_label_func, label_col, encode=False):
    df = df.dropna() # Drop Rows containing Nan as a label
    X = df.drop(label_col, axis=1)
    if encode:
        X = pd.get_dummies(X, drop_first=False)
    y = df[label_col]
    test_model_X_y(X, y, transform_label_func)

def test_model_X_y(X, y, transform_label_func):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    soft_model = SoftDecisionTreeClassifier()
    hard_model = DecisionTreeClassifier()
    print("Soft descision tree classifier:")
    train_and_eval(X_train, X_test, y_train, y_test, soft_model, transform_label_func)
    print("Regular descision tree classifier:")
    train_and_eval(X_train, X_test, y_train, y_test, hard_model, transform_label_func)

    

def train_and_eval(X_train, X_test, y_train, y_test, model, transform_label_func):
    model.fit(X_train.values, y_train.values)
    probas = model.predict_proba(X_test.to_numpy())
    y_predict = np.argmax(probas, axis=1)
    clean_y_test = transform_label_func(y_test.to_numpy())

    print("Accuracy")
    print(accuracy_score(clean_y_test, y_predict))
    
    # Need to sort out multiclass AUCs
    # print("AUC:")
    # roc_auc_score(clean_y_test, y_predict)



# ---------- Dataset 1: Graduation -----------

def test_model_graduation(dataset):
    print(" ----- Dataset 1 - Graduation -----")
    test_model_UCI(dataset, transform_label_graduation)
     
def transform_label_graduation(lst):
    return np.vectorize({"Dropout": 0, "Enrolled": 1, "Graduate": 2}.get)(lst)

# ----------- Dataset 2: mushrooms -----------

def test_model_mushrooms(dataset):
    print(" ----- Dataset 2 - mushrooms -----")
    test_model_UCI(dataset, transform_label_mushrooms, encode=True)


def transform_label_mushrooms(lst):
    return np.vectorize({"e": 0, "p": 1}.get)(lst)

# ----------- Dataset 3: Android -----------

def test_model_android(dataset):
    print(" ----- Dataset 3 - Android -----")
    test_model_csv(dataset, transform_label_android, label_col='Label')


def transform_label_android(lst):
    return lst



def main():
    # Dataset 1
    predict_students_dropout_and_academic_success = fetch_ucirepo(id=697)
    test_model_graduation(predict_students_dropout_and_academic_success)
    
    # Dataset 2
    secondary_mushroom = fetch_ucirepo(id=848)
    test_model_mushrooms(secondary_mushroom)

    # Dataset 3
    tunadromd = pd.read_csv("TUANDROMD.csv")
    test_model_android(tunadromd)

    # mushroom = fetch_ucirepo(id=73)
    # test_model(tunadromd)

    #mnist = 1

if __name__ == "__main__":
    main()