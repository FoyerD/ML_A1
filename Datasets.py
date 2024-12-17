
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.tree import _tree
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc
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
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import LabelBinarizer



from SoftDecisionTreeClassifier import SoftDecisionTreeClassifier
from SoftDecisionTreeRegression import SoftDecisionTreeRegressor

n = 100
a = 0.05
param_grid_test = {"n_runs": [10], "alpha": [0.00001]}
param_grid = {"n_runs": [10, 50], "alpha": [0.00001,0.01, 0.05]}
mega_param_grid = {'n_runs': [10, 50, 100], 'alpha': [0.001, 0.01, 0.05]}

# ---------- Classification ----------

def test_model_UCI(dataset, transform_label_func, encode=False, reverse_transform_label_func=None):
    X = dataset.data.features
    if encode:
        X = pd.get_dummies(X, drop_first=False)
    y = dataset.data.targets
    test_model_X_y(X, y, transform_label_func, reverse_transform_label_func)

def test_model_csv(df, transform_label_func, label_col, encode=False, reverse_transform_label_func=None):
    df = df.dropna() # Drop Rows containing Nan as a label
    X = df.drop(label_col, axis=1)
    if encode:
        X = pd.get_dummies(X, drop_first=False)
    y = df[label_col]
    test_model_X_y(X, y, transform_label_func, reverse_transform_label_func)

def test_model_X_y(X, y, transform_label_func, reverse_transform_label_func=None):
    y = transform_label_func(y.to_numpy())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    hard_model = DecisionTreeClassifier()
    soft_model = GridSearchCV(SoftDecisionTreeClassifier(), param_grid_test, cv=2)
    
    # soft_model = SoftDecisionTreeClassifier(n_runs=n, alpha=a)
    # hard_model = DecisionTreeClassifier(n_runs=n, alpha=a)
    classes = np.unique(y)

    print("Soft descision tree classifier:")
    y_pred_soft = train_and_eval(X_train, X_test, y_train, y_test, soft_model, classes)
    cross_val_clf(X_train, y_train, soft_model, 10)
    print("parameters: " + str(soft_model.best_params_))
    print("Regular descision tree classifier:")
    y_pred_hard = train_and_eval(X_train, X_test, y_train, y_test, hard_model, classes)
    cross_val_clf(X_train, y_train, hard_model, 10)

    if(len(classes) <= 2):
        plot_roc_curve(y_test, y_pred_soft, y_pred_hard)
    else:
        pass
        #plot_roc_multi_curve(y_test,y_pred_soft, y_pred_hard, reverse_transform_label_func, list(classes))

    confusion_matrix_plot(y_test, y_pred_soft, y_pred_hard)


def train_and_eval(X_train, X_test, y_train, y_test, model, classes):
    model.fit(X_train.values, y_train)
    probas = model.predict_proba(X_test.to_numpy())
    y_predict = np.argmax(probas, axis=1)

    print("Accuracy")
    print(accuracy_score(y_test, y_predict))
    
    if(len(classes) > 2):
        label_binarizer = LabelBinarizer().fit(y_test)
        y_onehot_test = label_binarizer.transform(y_test)
        print(y_onehot_test.shape)
        fpr, tpr, roc_auc = dict(), dict(), dict()
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), probas.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        print(f"Micro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['micro']:.2f}")

    else:
        fpr, tpr, thresholds = roc_curve(y_test, y_predict)
        print("AUC:")
        print(auc(fpr, tpr))

    return y_predict


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
    
def plot_roc_multi_curve(y_test, y_pred_soft, y_pred_hard, reverse_transform, classes):
    y_test = reverse_transform(y_test)
    y_pred_soft = reverse_transform(y_pred_soft)
    y_pred_hard = reverse_transform(y_pred_hard)

    y_onehot_test = y_test.transform(y_test)
    for class_of_interest in classes:
        class_id = np.flatnonzero(y_test.classes_ == class_of_interest)[0]

        display = RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_pred_soft[:, class_id],
            name=f"{class_of_interest} vs the rest",
            color="darkorange",
            plot_chance_level=True,
            despine=True,
        )   
        _ = display.ax_.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title="Soft {class_of_interest}-vs-Rest ROC curves",
        )

        display = RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_pred_soft[:, class_id],
            name=f"{class_of_interest} vs the rest",
            color="darkorange",
            plot_chance_level=True,
            despine=True,
        )   
        _ = display.ax_.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"Regular {class_of_interest}-vs-Rest ROC curves",
        )

def plot_roc_curve(y_test, y_pred_soft, y_pred_hard):
    fpr_soft, tpr_soft, _ = roc_curve(y_test, y_pred_soft)
    fpr_hard, tpr_hard, _ = roc_curve(y_test, y_pred_hard)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

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

    plt.tight_layout()
    plt.show()


def cross_val_clf(X, y, clf, f):
    print("Cross Validation:")
    scores = cross_val_score(clf, X, y, cv=f)
    print("%f accuracy with a standard deviation of %f" % (scores.mean(), scores.std()))


def confusion_matrix_plot(y_test, y_pred_soft, y_pred_hard):
    cm_soft = confusion_matrix(y_test, y_pred_soft)
    cm_hard = confusion_matrix(y_test, y_pred_hard)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_soft)
    disp1.plot(ax=axes[0])
    axes[0].set_title("Soft Decision Tree Classifier Confusion Matrix")

    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_hard)
    disp2.plot(ax=axes[1])
    axes[1].set_title("Regular Decision Tree Classifier Confusion Matrix")

    plt.tight_layout()

    plt.show()







# ---------- Regression ----------

def test_model_UCI_reg(dataset, encode=False):
    X = dataset.data.features
    if encode:
        X = pd.get_dummies(X, drop_first=False)
    y = dataset.data.targets
    test_model_X_y_reg(X, y)
    plot_target_distrubtion(y)

def test_model_csv_reg(df, label_col, encode=False):
    df = df.dropna() # Drop Rows containing Nan as a label
    X = df.drop(label_col, axis=1)
    if encode:
        X = pd.get_dummies(X, drop_first=False)
    y = df[label_col]
    test_model_X_y_reg(X, y)
    plot_target_distrubtion(y)

def test_model_X_y_reg(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    soft_model = SoftDecisionTreeRegressor()
    #soft_model = GridSearchCV(SoftDecisionTreeRegressor(), mega_param_grid, cv=5)
    hard_model = DecisionTreeRegressor()
    print("Soft descision tree regressor:")
    y_soft_pred = train_and_eval_reg(X_train, X_test, y_train, y_test, soft_model)
#    print("parameters: " + str(soft_model.best_params_))
    print("Regular descision tree regressor:")
    y_hard_pred = train_and_eval_reg(X_train, X_test, y_train, y_test, hard_model)
    plot_regression_graph(y_test, y_soft_pred, y_hard_pred)


    

def train_and_eval_reg(X_train, X_test, y_train, y_test, model):
    model.fit(X_train.values, y_train.values)
    y_predict = model.predict(X_test.to_numpy())

    print("RMSE:")
    print(rmse(y_test, y_predict))
    return y_predict

def plot_target_distrubtion(vals):
    n = len(vals)
    indecies = range(n)
    plt.scatter(indecies, vals)
    plt.xlabel("Index")
    plt.ylabel("Values")
    plt.title("Target Values Distribution")
    plt.show()

def plot_regression_graph(y_test, y_soft_pred, y_hard_pred):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].scatter(y_test, y_soft_pred, alpha=0.7)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
    axes[0].set_xlabel("True Values")
    axes[0].set_ylabel("Predicted Values")
    axes[0].set_title("Soft Regression Tree: True vs Predicted")

    axes[1].scatter(y_test, y_hard_pred, alpha=0.7)
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
    axes[1].set_xlabel("True Values")
    axes[1].set_ylabel("Predicted Values")
    axes[1].set_title("Regular Regression Tree: True vs Predicted")


    plt.tight_layout()
    plt.show()


# ---------- Dataset 1: Water quality -----------

#def test_model_water(dataset):
#    print(" ----- Dataset 1 - Water quality -----")
#    test_model_csv(dataset, transform_label_water, label_col="Potability", encode=True)
#    print("")
     
#def transform_label_water(lst):
#    return lst

# ---------- Dataset 1: Graduation -----------

def test_model_graduation(dataset):
    print(" ----- Dataset 1 - Graduation -----")
    test_model_UCI(dataset, transform_label_graduation, reverse_transform_label_graduation)

def transform_label_graduation(lst):
    return np.vectorize({"Dropout": 0, "Enrolled": 1, "Graduate": 2}.get)(lst)

def reverse_transform_label_graduation(lst):
    return np.vectorize({0 : "Dropout", 1: "Enrolled", 2: "Graduate"}.get)(lst)

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
    print(" ----- Dataset 5 - Mountain vs Beaches-----")
    test_model_csv(dataset, transform_label_mountain_vs_beaches, label_col='Preference', encode=True)
    print("")


def transform_label_mountain_vs_beaches(lst):
    return lst


def classification_data():
    
    # Dataset 1
    #water = pd.read_csv("water_potability.csv")
    #test_model_water(water)
    
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


def regression_data():

    # Dataset Reg 1
    print(" ----- Dataset 1 Reg - Wine quality -----")
    wine_quality = fetch_ucirepo(id=186) 
    test_model_UCI_reg(wine_quality)
    print("")

    # Dataset Reg 2
    print(" ----- Dataset 2 Reg - Diamond prices -----")
    diamonds = pd.read_csv("diamonds.csv")
    test_model_csv_reg(diamonds, label_col="price", encode=True)
    print("")

    # Dataset Reg 3
    print(" ----- Dataset 3 Reg - Units of alcohol in drinks -----")
    alcohol = pd.read_csv("alcohol.csv")
    test_model_csv_reg(alcohol, label_col="Units of Alcohol", encode=True)
    print("")

    # Dataset Reg 4
    print(" ----- Dataset 4 Reg - Movies worldwide gross -----")
    movies = pd.read_csv("movie_statistic_dataset.csv")
    test_model_csv_reg(movies, label_col="Worldwide gross $", encode=True)
    print("")

    # Dataset Reg 5
    print(" ----- Dataset 5 Reg - World wide happiness 2024 -----")
    happines = pd.read_csv("World_Happiness_Report_2024.csv")
    test_model_csv_reg(happines, label_col="Positive affect", encode=True)
    print("")


def main():
    regression_data()
    print("\n\n")
    #classification_data()

if __name__ == "__main__":
    main()