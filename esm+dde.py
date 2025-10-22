import numpy as np


# esm_features = np.load('esm_test_feature_80.npy')
# dde_features = np.load('dde-test-features.npy')
# esm_features_train = np.load('esm_feature_80.npy')
# dde_features_train = np.load('dde-features.npy')
esm_features = np.load('esm_test_feature_80.npy')
dde_features = np.load('dde_test_feature_80.npy')
esm_features_train = np.load('esm_feature_80.npy')
dde_features_train = np.load('dde_feature_80.npy')

print("ESM Features Shape:", esm_features.shape)
print("DDE Features Shape:", dde_features.shape)


assert esm_features.shape[0] == dde_features.shape[0], "same amount"


combined_features = np.concatenate((esm_features, dde_features), axis=1)
print("Combined Features Shape:", combined_features.shape)
combined_features_train = np.concatenate((esm_features_train, dde_features_train), axis=1)
print("Combined Features Shape:", combined_features_train.shape)

# np.save('combined-features-test-2.npy', combined_features)
# np.save('combined-features-train-2.npy', combined_features_train)
np.save('combined-features-test-3.npy', combined_features)
np.save('combined-features-train-3.npy', combined_features_train)