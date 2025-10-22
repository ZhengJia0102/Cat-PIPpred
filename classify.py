import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    StackingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import logging
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    matthews_corrcoef,
    average_precision_score,
    confusion_matrix,
    roc_curve,
)
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_evaluate_models():
    try:
        # X_train = np.load("train_feature_90.npy", allow_pickle=True)
        # X_test = np.load("test_feature_90.npy", allow_pickle=True)
        X_train = np.load('cksaap_combine_feature_80.npy', allow_pickle=True)
        X_test = np.load('cksaap_combine_test_feature_80.npy', allow_pickle=True)
        print(X_train)
        train_data = pd.read_excel(r"data\inflammatory_peptides_data\dd_0.xlsx")  
        test_data = pd.read_excel(r"data\inflammatory_peptides_data\dd.xlsx")
        y_train = train_data["label"]
        y_test = test_data["label"]
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

    
    imputer = SimpleImputer(strategy='mean')

    
    params = {
        "Decision Tree": {
            "criterion": "entropy",
            "max_depth": 6,
            "max_features": 'sqrt',
            "min_samples_leaf": 1,
            "min_samples_split": 5,
        },
        "Random Forest": {
            "criterion": "gini",
            "max_depth": 20,
            "max_features": "log2",
            "min_samples_leaf": 4,
            "min_samples_split": 10,
            "n_estimators": 300,
        },
        "AdaBoost": {"algorithm": "SAMME.R", "learning_rate": 0.5, "n_estimators": 200},
        "CatBoost": {
            "iterations": 100,
            "learning_rate": 0.1,
            "depth": 4,
            "l2_leaf_reg": 3,
            "border_count": 32,
            "random_seed": 42,
            "verbose": 0
        },
        "XGBoost": {
            "colsample_bytree": 0.6,
            "gamma": 0,
            "learning_rate": 0.05,
            "max_depth": 20,
            "min_child_weight": 1,
            "n_estimators": 200,
            "subsample": 1,
        }
    }

    
    models = {
        "Decision Tree": make_pipeline(imputer, DecisionTreeClassifier(**params["Decision Tree"])),
        "Random Forest": make_pipeline(imputer, RandomForestClassifier(**params["Random Forest"])),
        "AdaBoost": make_pipeline(imputer, AdaBoostClassifier(**params["AdaBoost"])),
        "CatBoost": make_pipeline(imputer, CatBoostClassifier(**params["CatBoost"])),
        "XGBoost": make_pipeline(imputer, XGBClassifier(**params["XGBoost"])),
        "Stacking": make_pipeline(
            imputer,
            StackingClassifier(
                estimators=[
                    ("XGBoost", XGBClassifier(**params["XGBoost"])),
                    ("CatBoost", CatBoostClassifier(**params["CatBoost"])),
                ],
                final_estimator=LogisticRegression(),
            ),
        ),
    }

    
    for name in models:
        logging.info(f"{name} is training...")
        models[name].fit(X_train, y_train)

    
    test_metrics = {}
    test_roc = {}
    roc_flag = True  

    
    plt.figure()    

    
    count = 0  

    
    custom_colors = ['#FDDED7', '#F5BE8F', '#C1E0DB', '#CCD376', '#A28CC2','#5CB0C3']

    
    if len(models) > len(custom_colors):
        raise ValueError("warning")

    
    for name in models:
        logging.info(f"{name} is testing...")
        cr = {}
        model = models[name]
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        cm = confusion_matrix(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        cr["AUC"] = auc
        cr["Acc"] = (TN + TP) / (TN + FP + FN + TP)
        cr["MCC"] = ((TP * TN) - (FP * FN)) / (
            np.sqrt((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN))
        )
        cr["Sn"] = TP / (TP + FN)
        cr["Sp"] = TN / (TN + FP)
        cr["CM"] = str(cm.tolist())
        test_metrics[name] = cr

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        test_roc[name] = (auc, fpr, tpr)

        
        if roc_flag:
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})', color=custom_colors[count])
            count += 1

    
    plt.plot([0, 1], [0, 1], 'k--')

    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Models')
    plt.legend(loc='lower right')

    
    if roc_flag:
        plt.savefig('roc_curves_all_models.png')
        plt.show()

    
    save_results_to_csv(test_metrics)

    if roc_flag:
        return test_metrics, test_roc
    else:
        return test_metrics

def save_results_to_csv(results):
    df = pd.DataFrame.from_dict(results, orient="index")
    df.to_csv("esm_cksaap_results.csv", index_label="Model")
    logging.info("Results saved to model_metrics_four_Fusion_combine.csv")


results = train_and_evaluate_models()
if results:
    print(results)