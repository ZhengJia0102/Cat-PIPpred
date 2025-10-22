import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


X_train = np.load('cksaap-features.npy', allow_pickle=True)
X_test = np.load('cksaap-test-features.npy', allow_pickle=True)  
train_data = pd.read_excel(r'data\inflammatory_peptides_data\dd_0.xlsx')  
test_data = pd.read_excel(r'data\inflammatory_peptides_data\dd.xlsx')
y_train = train_data['label']
y_test = test_data['label']


imputer = SimpleImputer(strategy='mean')  
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)


lasso = LassoCV(cv=10, random_state=42)
lasso.fit(X_train_scaled, y_train)


selected_features_lasso = np.where(lasso.coef_ != 0)[0]
print("Lasso :", len(selected_features_lasso))


X_train_selected = X_train_scaled[:, selected_features_lasso]
X_test_selected = X_test_scaled[:, selected_features_lasso]


# np.save('esm_feature_80.npy', X_train_selected)
# np.save('esm_test_feature_80.npy', X_test_selected)
# np.save('combine_feature_80.npy', X_train_selected)
# np.save('combine_test_feature_80.npy', X_test_selected)
#np.save('dde_feature_80.npy', X_train_selected)
#np.save('dde_test_feature_80.npy', X_test_selected)
np.save('cksaap_combine_feature_80.npy', X_train_selected)
np.save('cksaap_combine_test_feature_80.npy', X_test_selected)
print("finished")
