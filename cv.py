import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    f1_score
)
import logging
import os
from datetime import datetime
import math


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def calculate_sensitivity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def calculate_accuracy_from_cm(tn, fp, fn, tp):
    total = tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else 0

def calculate_mcc_from_cm(tn, fp, fn, tp):
    numerator = (tp * tn) - (fp * fn)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / denominator if denominator != 0 else 0

def load_data():
    try:
        #features = np.load('stage-2-features_2.npy', allow_pickle=True)
        features = np.load('stage-2-features_3.npy', allow_pickle=True)
        label_data = pd.read_excel("data/inflammatory_peptides_data/dd_0.xlsx")
        return features, label_data["label"]
    except Exception as e:
        logging.error(f"fail: {e}")
        raise

def get_model_configurations():
    return {
        "Decision Tree": {
            "model": DecisionTreeClassifier(
                criterion="entropy",
                max_depth=6,
                max_features='sqrt',
                min_samples_leaf=1,
                min_samples_split=5
            )
        },
        "Random Forest": {
            "model": RandomForestClassifier(
                criterion="gini",
                max_depth=20,
                max_features="log2",
                min_samples_leaf=4,
                min_samples_split=10,
                n_estimators=300
            )
        },
        "AdaBoost": {
            "model": AdaBoostClassifier(
                algorithm="SAMME.R", 
                learning_rate=0.5, 
                n_estimators=200
            )
        },
        "CatBoost": {
            "model": CatBoostClassifier(
                iterations=100,
                learning_rate=0.1,
                depth=4,
                l2_leaf_reg=3,
                border_count=32,
                random_seed=42,
                verbose=0
            )
        },
        "XGBoost": {
            "model": XGBClassifier(
                colsample_bytree=0.6,
                gamma=0,
                learning_rate=0.05,
                max_depth=20,
                min_child_weight=1,
                n_estimators=200,
                subsample=1
            )
        },
        "Stacking": {
            "model": StackingClassifier(
                estimators=[
                    ("XGBoost", XGBClassifier(
                        colsample_bytree=0.6,
                        gamma=0,
                        learning_rate=0.05,
                        max_depth=20,
                        min_child_weight=1,
                        n_estimators=200,
                        subsample=1
                    )),
                    ("CatBoost", CatBoostClassifier(
                        iterations=100,
                        learning_rate=0.1,
                        depth=4,
                        l2_leaf_reg=3,
                        border_count=32,
                        random_seed=42,
                        verbose=0
                    )),
                ],
                final_estimator=LogisticRegression()
            )
        }
    }

def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None
    

    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    
    metrics = {
        "Accuracy": calculate_accuracy_from_cm(tn, fp, fn, tp),
        "F1": f1_score(y_val, y_pred),
        "MCC": calculate_mcc_from_cm(tn, fp, fn, tp),
        "Sn": calculate_sensitivity(y_val, y_pred),
        "Sp": calculate_specificity(y_val, y_pred)
    }
    
    if y_prob is not None:
        metrics["AUC"] = roc_auc_score(y_val, y_prob)
    
    return metrics

def save_results_to_csv(results, filename="esm_cksaap_cv2_results.csv"):
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(results_dir, f"{timestamp}_{filename}")
    
    
    df = pd.DataFrame(results)
    
    
    df.to_csv(filepath, index=False)
    logging.info(f"save: {filepath}")
    return filepath

def train_and_evaluate():
    try:
        X, y = load_data()
        imputer = SimpleImputer(strategy='mean')
        models_config = get_model_configurations()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=36)
        
        
        all_results = []
        
        for model_name, config in models_config.items():
            logging.info(f"evaluate: {model_name}")
            model_pipeline = make_pipeline(imputer, config["model"])
            fold_results = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model_pipeline.fit(X_train, y_train)
                metrics = evaluate_model(model_pipeline, X_val, y_val)
                
                
                fold_result = {
                    "Model": model_name,
                    "Fold": fold_idx,
                    **metrics
                }
                all_results.append(fold_result)
                
                logging.info(
                    f"Fold {fold_idx}: "
                    f"Acc={metrics['Accuracy']:.4f}, "
                    f"AUC={metrics.get('AUC', 0):.4f}, "
                    f"MCC={metrics['MCC']:.4f}, "
                    f"Sn={metrics['Sn']:.4f}, "
                    f"Sp={metrics['Sp']:.4f}"
                )
                fold_results.append(metrics)
            
            
            avg_metrics = {
                "Model": model_name,
                "Fold": "Average",
            }
            for metric in fold_results[0].keys():
                avg_metrics[metric] = np.mean([fold[metric] for fold in fold_results])
            
            all_results.append(avg_metrics)
            logging.info(f"{model_name} Average: "
                         f"Acc={avg_metrics['Accuracy']:.4f}, "
                         f"AUC={avg_metrics.get('AUC', 0):.4f}, "
                         f"Sn={avg_metrics['Sn']:.4f}, "
                         f"Sp={avg_metrics['Sp']:.4f}, "
                         f"MCC={avg_metrics['MCC']:.4f}")
        
        
        save_results_to_csv(all_results)
        return all_results
    
    except Exception as e:
        logging.error(f"warning: {e}")
        return None

if __name__ == "__main__":
    model_results = train_and_evaluate()
    
    if model_results:
        results_df = pd.DataFrame(model_results)
        
        logging.info("\nsummary:")
        for model_name in results_df['Model'].unique():
            model_data = results_df[results_df['Model'] == model_name]
            avg_data = model_data[model_data['Fold'] == 'Average']
            
            logging.info(f"\n{model_name} Average:")
            for metric in ['Accuracy', 'MCC', 'Sn', 'Sp', 'AUC']:
                if metric in avg_data.columns:
                    value = avg_data[metric].values[0]
                    logging.info(f"  {metric}: {value:.4f}")