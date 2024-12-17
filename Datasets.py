import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import _tree
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import root_mean_squared_error as rmse
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import PrecisionRecallDisplay, accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import mean_squared_error as mse
import pandas as pd
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt


from SoftDecisionTreeClassifier import SoftDecisionTreeClassifier
from SoftDecisionTreeRegression import SoftDecisionTreeRegressor

# ---------- Classification ----------

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
    
    #extractin metrics
    #clean_y_test = clean_y_test.reshape((clean_y_test.shape[0]))
    #y_predict = y_predict.reshape((y_predict.shape[0], 1))
    fpr, tpr, thresholds = roc_curve(clean_y_test, y_predict)
    print("AUC:")
    print(auc(fpr, tpr))


def plot_pr_curve(X_test, y_test, soft_clf, hard_clf):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # First plot
    display1 = PrecisionRecallDisplay.from_estimator(
        soft_clf, X_test, y_test, name="SoftDecisionTreeClassifier", plot_chance_level=True, ax=axes[0]
    )
    axes[0].set_title("Soft Decision Tree Classifier")

    # Second plot
    display2 = PrecisionRecallDisplay.from_estimator(
        hard_clf, X_test, y_test, name="DecsionTreeClassifier", plot_chance_level=True, ax=axes[1]
    )
    axes[1].set_title("Regular Decision Tree Classifier")

    plt.tight_layout()
    plt.show()
    

def plot_roc_curve(y, y_pred_soft, y_pred_hard):
    fpr_soft, tpr_soft, _ = roc_curve(y, y_pred_soft)
    fpr_hard, tpr_hard, _ = roc_curve(y, y_pred_hard)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Soft decision tree
    axes[0].plot(fpr_soft, tpr_soft)
    axes[0].set_title("Soft Decision Tree Classifier")
    axes[0].xlabel('False Positive Rate')
    axes[0].ylabel('True Positive Rate')

    # Regular decision tree
    axes[1].plot(fpr_hard, tpr_hard)
    axes[1].set_title("Regular Decision Tree Classifier")
    axes[1].xlabel('False Positive Rate')
    axes[1].ylabel('True Positive Rate')

    plt.tight_layout()
    plt.show()


def cross_val_clf(X, y, clf, f):
    scores = cross_val_score(clf, X, y, cv=f)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))



# ---------- Regression ----------

def test_model_UCI_reg(dataset, encode=False):
    X = dataset.data.features
    if encode:
        X = pd.get_dummies(X, drop_first=False)
    y = dataset.data.targets
    test_model_X_y_reg(X, y)

def test_model_csv_reg(df, label_col, encode=False):
    df = df.dropna() # Drop Rows containing Nan as a label
    X = df.drop(label_col, axis=1)
    if encode:
        X = pd.get_dummies(X, drop_first=False)
    y = df[label_col]
    test_model_X_y_reg(X, y)

def test_model_X_y_reg(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    soft_model = SoftDecisionTreeRegressor()
    hard_model = DecisionTreeRegressor()
    print("Soft descision tree regressor:")
    train_and_eval_reg(X_train, X_test, y_train, y_test, soft_model)
    print("Regular descision tree regressor:")
    train_and_eval_reg(X_train, X_test, y_train, y_test, hard_model)

    

def train_and_eval_reg(X_train, X_test, y_train, y_test, model):
    model.fit(X_train.values, y_train.values)
    y_predict = model.predict(X_test.to_numpy())

    print("RMSE:")
    print(rmse(y_test, y_predict))

# ---------- Dataset 1: Apples -----------

def test_model_water(dataset):
    print(" ----- Dataset 1 - Water quality -----")
    test_model_csv(dataset, transform_label_water, label_col="Potability", encode=True)
    print("")
     
def transform_label_water(lst):
    return lst

# ----------- Dataset 2: mushrooms -----------

def test_model_mushrooms(dataset):
    print(" ----- Dataset 2 - mushrooms -----")
    test_model_UCI(dataset, transform_label_mushrooms, encode=True)
    print("")


def transform_label_mushrooms(lst):
    return np.vectorize({"e": 0, "p": 1}.get)(lst)

# ----------- Dataset 3: Android -----------

def test_model_android(dataset):
    print(" ----- Dataset 3 - Android -----")
    test_model_csv(dataset, transform_label_android, label_col='Label')
    print("")


def transform_label_android(lst):
    return lst

# ----------- Dataset 4: Gym Membership -----------
def test_model_gym(dataset):
    print(" ----- Dataset 4 - Gym Membership -----")
    test_model_csv(dataset, transform_label_gym, label_col='Gender', encode=True)
    print("")


def transform_label_gym(lst):
    return np.vectorize({"Male": 1, "Female": 0}.get)(lst)

# ----------- Dataset 5: Mountans vs Beaches -----------
def test_model_mountain_vs_beaches(dataset):
    print(" ----- Dataset 5 - Mountain vs Beaches--")
    test_model_csv(dataset, transform_label_mountain_vs_beaches, label_col='Preference', encode=True)
    print("")


def transform_label_mountain_vs_beaches(lst):
    return lst



def classification_data():
    
    # Dataset 1
    water = pd.read_csv("water_potability.csv")
    test_model_water(water)
    
    # Dataset 2
    secondary_mushroom = fetch_ucirepo(id=848)
    test_model_mushrooms(secondary_mushroom)

    # Dataset 3
    tunadromd = pd.read_csv("TUANDROMD.csv")
    test_model_android(tunadromd)

    # Dataset 4
    gym_membership = pd.read_csv("gym_members_exercise_tracking.csv")
    test_model_gym(gym_membership)

    # Dataset 5
    mountains_vs_beaches = pd.read_csv("mountains_vs_beaches_preferences.csv")
    test_model_mountain_vs_beaches(mountains_vs_beaches)

def regression_data():

    # Dataset Reg 1
    print(" ----- Dataset 1 Reg - Wine quality --")
    wine_quality = fetch_ucirepo(id=186) 
    test_model_UCI_reg(wine_quality)
    print("")

    # Dataset Reg 2
    print(" ----- Dataset 2 Reg - Diamond prices --")
    diamonds = pd.read_csv("diamonds.csv")
    test_model_csv_reg(diamonds, label_col="price", encode=True)
    print("")

    # Dataset Reg 3
    print(" ----- Dataset 3 Reg - Units of alcohol in drinks --")
    alcohol = pd.read_csv("alcohol.csv")
    test_model_csv_reg(alcohol, label_col="Units of Alcohol", encode=True)
    print("")

    # Dataset Reg 4
    print(" ----- Dataset 4 Reg - Movies worldwide gross --")
    movies = pd.read_csv("movie_statistic_dataset.csv")
    test_model_csv_reg(movies, label_col="Worldwide gross $", encode=True)
    print("")

    # Dataset Reg 5
    print(" ----- Dataset 5 Reg - World wide happiness 2024 --")
    happines = pd.read_csv("World_Happiness_Report_2024.csv")
    test_model_csv_reg(happines, label_col="Positive affect", encode=True)
    print("")


def main():
    regression_data()
    print("\n\n")
    classification_data()

if __name__ == "__main__":
    main()