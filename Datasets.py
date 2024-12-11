import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import _tree
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import mean_squared_error as mse
import pandas as pd
from ucimlrepo import fetch_ucirepo

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
    
    # Need to sort out multiclass AUCs
    # print("AUC:")
    # roc_auc_score(clean_y_test, y_predict)

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

    print("MSE:")
    print(mse(y_test, y_predict))

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


def classification_data():
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

def regression_data():
    wine_quality = fetch_ucirepo(id=186) 
    test_model_UCI_reg(wine_quality)


def main():
    regression_data()
    classification_data()

if __name__ == "__main__":
    main()