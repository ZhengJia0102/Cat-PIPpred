import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import shap


X = np.load('combine_feature_80.npy', allow_pickle=True)
y_data = pd.read_excel("data/inflammatory_peptides_data/dd_0.xlsx")
y = y_data['label'].values  


model = make_pipeline(
    SimpleImputer(strategy='mean'),
    CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=4,
        l2_leaf_reg=3,
        border_count=32,
        random_seed=42,
        verbose=0
    )
)

model.fit(X, y)


final_model = model.named_steps['catboostclassifier']
X_processed = model.named_steps['simpleimputer'].transform(X)


explainer = shap.TreeExplainer(final_model)


shap_values = explainer.shap_values(X_processed)


if isinstance(shap_values, list):
    if len(shap_values) > 1:
        shap_values = shap_values[1]  
    else:
        shap_values = shap_values[0]
elif len(shap_values.shape) == 3:
    shap_values = shap_values[:, :, 1]


feature_names = [f'feature_{i}' for i in range(X_processed.shape[1])]


try:
    shap.summary_plot(shap_values, X_processed, feature_names=feature_names)
except Exception as e:
    print(f"warning: {e}")
    shap.summary_plot(shap_values, X_processed, feature_names=feature_names, plot_type="bar")