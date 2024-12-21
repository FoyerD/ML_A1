
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, RepeatedKFold
from sklearn.tree import _tree
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import root_mean_squared_error as rmse
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.metrics import PrecisionRecallDisplay, accuracy_score, roc_curve
import pandas as pd
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import LabelBinarizer



from SoftDecisionTreeClassifier import SoftDecisionTreeClassifier
from SoftDecisionTreeRegression import SoftDecisionTreeRegressor
from WeightedDecisionTreeClassifier import WeightedSoftDecisionTreeClassifier

param_grid_test = {"n_runs": [100], "alpha": [0.01]}
param_grid = {"n_runs": [10, 50], "alpha": [0.01, 0.05,0.1]}
mega_param_grid = {'n_runs': [10, 50, 100], 'alpha': [0.01, 0.05,0.1]}

# ---------- Classification ----------

def test_model_UCI(dataset, transform_label_func, encode=False, ds_name=None):
    X = dataset.data.features
    if encode:
        X = pd.get_dummies(X, drop_first=False)
    y = dataset.data.targets
    test_model_X_y(X, y, transform_label_func,ds_name)

def test_model_csv(df, transform_label_func, label_col, encode=False, ds_name=None):
    df = df.dropna() # Drop Rows containing Nan as a label
    df = df.sample(frac = 1)
    X = df.drop(label_col, axis=1)
    if encode:
        X = pd.get_dummies(X, drop_first=False)
    y = df[label_col]
    test_model_X_y(X, y, transform_label_func, ds_name)

def test_model_X_y(X, y, transform_label_func, ds_name):
    y = transform_label_func(y.to_numpy())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Testing Adaptive tree
    hard_model = DecisionTreeClassifier()
    weighted_model = WeightedSoftDecisionTreeClassifier()
    soft_model = SoftDecisionTreeClassifier()

    classes = np.unique(y)
    print("Soft descision tree classifier:")
    y_pred_soft = train_and_eval(X_train, X_test, y_train, y_test, soft_model, classes)
    cross_val_clf(X_train, y_train, soft_model, 10)
    grid_search(X_train,y_train,X_test,y_test ,SoftDecisionTreeClassifier() , mega_param_grid, classes)
    print("\nWeighted descision tree classifier:")
    y_pred_weight = train_and_eval(X_train, X_test, y_train, y_test, weighted_model, classes)
    cross_val_clf(X_train, y_train, weighted_model, 10)
    grid_search(X_train,y_train,X_test,y_test,WeightedSoftDecisionTreeClassifier() , mega_param_grid,classes)
    print("\nRegular descision tree classifier:")
    y_pred_hard = train_and_eval(X_train, X_test, y_train, y_test, hard_model, classes)
    cross_val_clf(X_train, y_train, hard_model, 10)

    if(len(classes) <= 2):
        plot_roc_curve(y_test, y_pred_soft, y_pred_hard, y_pred_weight, ds_name=ds_name)
    else:
        pass #Not working
        #plot_roc_multi_curve(y_test,y_pred_soft, y_pred_hard, reverse_transform_label_func, list(classes))

    confusion_matrix_plot(y_test, y_pred_soft, y_pred_hard, y_pred_weight, ds_name=ds_name)

def grid_search(X_train,y_train, X_test, y_test,model, grid, classes, cv=10):
    if(len(classes) > 2):
        scoring = "roc_auc_ovr"
        temp_str = " Micro Avareged OvR multicalss"
    else:
        scoring = "roc_auc"
        temp_str = ""

    gscv = GridSearchCV(model, grid, cv=cv,scoring=scoring)

    gscv.fit(X_train.values, y_train)
    print("GS best parameters: " + str(gscv.best_params_))
    best_estimator = gscv.best_estimator_
    probas = best_estimator.predict_proba(X_test.to_numpy())
    y_predict = np.argmax(probas, axis=1)
    
    print("GS Accuracy: " + str(accuracy_score(y_test, y_predict)))

    if(len(classes) > 2):
        label_binarizer = LabelBinarizer().fit(y_test)
        y_test = label_binarizer.transform(y_test)
        y_predict = label_binarizer.transform(y_predict)
    fpr, tpr, _ = roc_curve(y_test.ravel(), y_predict.ravel())
    print("GS AUC" + temp_str + ": " + str(auc(fpr, tpr)))
        

def train_and_eval(X_train, X_test, y_train, y_test, model, classes):
    model.fit(X_train.values, y_train)
    probas = model.predict_proba(X_test.to_numpy())
    y_predict = np.argmax(probas, axis=1)

    print("Accuracy: " + str(accuracy_score(y_test, y_predict)))

    if(len(classes) > 2):
        temp_str = " Micro Avareged OvR multicalss"
        label_binarizer = LabelBinarizer().fit(y_test)
        y_test = label_binarizer.transform(y_test)
        y_predict_trans = label_binarizer.transform(y_predict)
    else:
        temp_str = ""
        y_predict_trans = y_predict
    fpr, tpr, _ = roc_curve(y_test.ravel(), y_predict_trans.ravel())
    print("AUC" + temp_str + ": " + str(auc(fpr, tpr)))

    return y_predict


def plot_pr_curve(X_test, y_test, soft_clf, hard_clf,weighted_clf, ds_name):
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

    # Third plot
    display3 = PrecisionRecallDisplay.from_estimator(
        weighted_clf, X_test, y_test, name="WeightedDecsionTreeClassifier", plot_chance_level=True, ax=axes[1]
    )
    axes[2].set_title("Weighted Decision Tree Classifier")

    plt.tight_layout()
    plt.savefig("plots/" + ds_name + "_pr.png")
    plt.clf()

def plot_roc_curve(y_test, y_pred_soft, y_pred_hard, y_pred_weighted, ds_name):
    fpr_soft, tpr_soft, _ = roc_curve(y_test, y_pred_soft)
    fpr_hard, tpr_hard, _ = roc_curve(y_test, y_pred_hard)
    fpr_weighted, tpr_weighted, _ = roc_curve(y_test, y_pred_weighted)

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

    # Soft decision tree
    axes[0].plot(fpr_soft, tpr_soft)
    axes[0].set_title("Soft Decision Tree Classifier")
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')

    # Regular decision tree
    axes[1].plot(fpr_hard, tpr_hard)
    axes[1].set_title("Regular Decision Tree Classifier")
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')

    axes[2].plot(fpr_weighted, tpr_weighted)
    axes[2].set_title("Weighted Decision Tree Classifier")
    axes[2].set_xlabel('False Positive Rate')
    axes[2].set_ylabel('True Positive Rate')

    plt.tight_layout()
    plt.savefig("plots/" + ds_name + "_roc.png")
    plt.clf()


def cross_val_clf(X, y, clf, f):
    print("Cross Validation:")
    rkf = RepeatedKFold(n_splits=5, n_repeats=2)
    scores = rkf.get_n_splits(X=X,y=y)
    scores = cross_val_score(clf, X, y, cv=rkf)
    print("%f accuracy with a standard deviation of %f" % (scores.mean(), scores.std()))


def confusion_matrix_plot(y_test, y_pred_soft, y_pred_hard, y_pred_weight, ds_name):
    cm_soft = confusion_matrix(y_test, y_pred_soft)
    cm_hard = confusion_matrix(y_test, y_pred_hard)
    cm_weight = confusion_matrix(y_test, y_pred_weight)

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_soft)
    disp1.plot(ax=axes[0])
    axes[0].set_title("Soft Decision Tree Classifier")

    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_hard)
    disp2.plot(ax=axes[1])
    axes[1].set_title("Regular Decision Tree Classifier")

    disp3 = ConfusionMatrixDisplay(confusion_matrix=cm_weight)
    disp3.plot(ax=axes[2])
    axes[2].set_title("Weighted Decision Tree Classifier")

    plt.tight_layout()

    plt.savefig("plots/" + ds_name + "_cm.png")
    plt.clf()







# ---------- Regression ----------

def test_model_UCI_reg(dataset, encode=False, ds_name=None):
    X = dataset.data.features
    if encode:
        X = pd.get_dummies(X, drop_first=False)
    y = dataset.data.targets
    test_model_X_y_reg(X, y,ds_name)
    plot_target_distrubtion(y,ds_name)

def test_model_csv_reg(df, label_col, encode=False, ds_name=None):
    df = df.dropna() # Drop Rows containing Nan as a label
    X = df.drop(label_col, axis=1)
    if encode:
        X = pd.get_dummies(X, drop_first=False)
    y = df[label_col]
    test_model_X_y_reg(X, y,ds_name=ds_name)
    plot_target_distrubtion(y,ds_name)

def test_model_X_y_reg(X, y, ds_name=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    soft_model = SoftDecisionTreeRegressor()
    hard_model = DecisionTreeRegressor()

    print("Soft descision tree regressor:")
    y_soft_pred = train_and_eval_reg(X_train, X_test, y_train, y_test, soft_model)

    soft_gs = GridSearchCV(SoftDecisionTreeRegressor(), mega_param_grid, cv=10, scoring="neg_root_mean_squared_error")
    soft_gs.fit(X_train.values, y_train.values)
    best_estimator = soft_gs.best_estimator_
    y_predict = best_estimator.predict(X_test.to_numpy())
    print("GS best parameters: " + str(soft_gs.best_params_))
    print("GS RMSE: " + str(rmse(y_test, y_predict)))

    print("\nRegular descision tree regressor:")
    y_hard_pred = train_and_eval_reg(X_train, X_test, y_train, y_test, hard_model)
    
    plot_regression_graph(y_test, y_soft_pred, y_hard_pred, ds_name=ds_name)


    

def train_and_eval_reg(X_train, X_test, y_train, y_test, model):
    model.fit(X_train.values, y_train.values)
    y_predict = model.predict(X_test.to_numpy())

    print("RMSE: " + str(rmse(y_test, y_predict)))
    return y_predict

def plot_target_distrubtion(vals, ds_name):
    n = len(vals)
    indecies = range(n)
    plt.scatter(indecies, vals)
    plt.xlabel("Index")
    plt.ylabel("Values")
    plt.title("Target Values Distribution " + ds_name)
    plt.savefig("plots/" + ds_name + "_reg_dist.png")
    plt.clf()

def plot_regression_graph(y_test, y_soft_pred, y_hard_pred, ds_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].scatter(y_test, y_soft_pred, alpha=0.7)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
    axes[0].set_xlabel("True Values")
    axes[0].set_ylabel("Predicted Values")
    axes[0].set_title("Soft Regression Tree: True vs Predicted " + ds_name)

    axes[1].scatter(y_test, y_hard_pred, alpha=0.7)
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
    axes[1].set_xlabel("True Values")
    axes[1].set_ylabel("Predicted Values")
    axes[1].set_title("Regular Regression Tree: True vs Predicted " + ds_name)


    plt.tight_layout()
    plt.savefig("plots/" + ds_name + "_reg.png")
    plt.clf()



# ---------- Dataset 1: Graduation -----------

def test_model_graduation(dataset):
    print(" ----- Dataset 1 - Graduation -----")
    test_model_UCI(dataset, transform_label_graduation, encode=True,ds_name="graduation")

def transform_label_graduation(lst):
    return np.vectorize({"Dropout": 0, "Enrolled": 1, "Graduate": 2}.get)(lst)

def reverse_transform_label_graduation(lst):
    return np.vectorize({0 : "Dropout", 1: "Enrolled", 2: "Graduate"}.get)(lst)

# ----------- Dataset 2: mushrooms -----------

def test_model_mushrooms(dataset):
    print(" ----- Dataset 2 - mushrooms -----")
    test_model_UCI(dataset, transform_label_mushrooms, encode=True,ds_name="mushrooms")
    print("")


def transform_label_mushrooms(lst):
    return np.vectorize({"e": 0, "p": 1}.get)(lst)

# ----------- Dataset 3: Android -----------

def test_model_android(dataset):
    print(" ----- Dataset 3 - Android -----")
    test_model_csv(dataset, transform_label_android, label_col='Label',ds_name="android")
    print("")


def transform_label_android(lst):
    return lst

# ----------- Dataset 4: Gym Membership -----------
def test_model_gym(dataset):
    print(" ----- Dataset 4 - Gym Membership -----")
    test_model_csv(dataset, transform_label_gym, label_col='Gender', encode=True,ds_name="gym_membership")
    print("")


def transform_label_gym(lst):
    return np.vectorize({"Male": 1, "Female": 0}.get)(lst)

# ----------- Dataset 5: Mountans vs Beaches -----------
def test_model_mountain_vs_beaches(dataset):
    print(" ----- Dataset 5 - Mountain vs Beaches-----")
    test_model_csv(dataset, transform_label_mountain_vs_beaches, label_col='Preference', encode=True,ds_name="mountain_vs_beaches")
    print("")


def transform_label_mountain_vs_beaches(lst):
    return lst


# ---------- Dataset 6: Water quality -----------

def test_model_water(dataset):
    print(" ----- Dataset 6 - Water quality -----")
    test_model_csv(dataset, transform_label_water, label_col="Potability", encode=True,ds_name="water_quality")
    print("")
     
def transform_label_water(lst):
    return lst


# ---------- Dataset 7: Wine quality -----------
def test_model_wine(dataset):
    print(" ----- Dataset 7 - Wine quality -----")
    dataset = dataset.drop
    test_model_UCI(dataset, transform_label_wine, encode=True,ds_name="wine_quality_class")
    print("")
     
def transform_label_wine(lst):
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

    # Dataset 4
    gym_membership = pd.read_csv("gym_members_exercise_tracking.csv")
    test_model_gym(gym_membership)

    # Dataset 5
    mountains_vs_beaches = pd.read_csv("mountains_vs_beaches_preferences.csv")
    test_model_mountain_vs_beaches(mountains_vs_beaches)

    # Dataset 6
    water = pd.read_csv("water_potability.csv")
    test_model_water(water)
    
    # Dataset 7
    #wine_quality = fetch_ucirepo(id=186)
    #test_model_wine(wine_quality)


def regression_data():

    # Dataset Reg 1
    print(" ----- Dataset 1 Reg - Wine quality -----")
    wine_quality = fetch_ucirepo(id=186)
    test_model_UCI_reg(wine_quality, encode=True,ds_name="wine_quality_reg")
    print("")

    # Dataset Reg 2
    print(" ----- Dataset 2 Reg - Diamond prices -----")
    diamonds = pd.read_csv("diamonds.csv")
    test_model_csv_reg(diamonds, label_col="price", encode=True,ds_name="diamond_prices")
    print("")

    # Dataset Reg 3
    print(" ----- Dataset 3 Reg - Units of alcohol in drinks -----")
    alcohol = pd.read_csv("alcohol.csv")
    test_model_csv_reg(alcohol, label_col="Units of Alcohol", encode=True,ds_name="alcohol_units")
    print("")

    # Dataset Reg 4
    print(" ----- Dataset 4 Reg - Movies worldwide gross -----")
    movies = pd.read_csv("movie_statistic_dataset.csv")
    test_model_csv_reg(movies, label_col="Worldwide gross $", encode=True,ds_name="movies_ww_gross")
    print("")

    # Dataset Reg 5
    print(" ----- Dataset 5 Reg - World wide happiness 2024 -----")
    happines = pd.read_csv("World_Happiness_Report_2024.csv")
    test_model_csv_reg(happines, label_col="Positive affect", encode=True,ds_name="ww_happiness")
    print("")


def main():
    regression_data()
    print("\n\n")
    classification_data()

if __name__ == "__main__":
    main()