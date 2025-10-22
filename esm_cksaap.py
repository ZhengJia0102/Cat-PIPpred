import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import esm
from collections import Counter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.to(device)  
model.eval()  


def extract_esm_features(sequence):
    batch_converter = alphabet.get_batch_converter()
    data = [("protein", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])  
    token_representations = results["representations"][33]
    
   
    sequence_representations = token_representations[:, 1:-1, :]
    
    avg_pool = sequence_representations.mean(dim=1)
    max_pool = sequence_representations.max(dim=1).values
    
    pooled_output = torch.cat((avg_pool, max_pool), dim=1).squeeze().cpu().numpy()
    return pooled_output


def extract_cksaap_features(fastas, gap=5):
    
    AA = 'ACDEFGHIKLMNPQRSTVWY'  # 20
    aaPairs = []
    for aa1 in AA:
        for aa2 in AA:
            aaPairs.append(aa1 + aa2)
    
    features = []
    
    for name, sequence in fastas:
        sequence = sequence.upper()  
        feature_vector = []
        
        for g in range(gap + 1):
            myDict = {pair: 0 for pair in aaPairs}
            sum_count = 0
            
            for index1 in range(len(sequence)):
                index2 = index1 + g + 1
                if index2 < len(sequence):
                    pair = sequence[index1] + sequence[index2]
                    if pair in myDict:
                        myDict[pair] += 1
                        sum_count += 1
            
            
            for pair in aaPairs:
                frequency = myDict[pair] / sum_count if sum_count > 0 else 0
                feature_vector.append(frequency)
        
        features.append(feature_vector)
    
    return np.array(features)


train_data_path = r'data\inflammatory_peptides_data\dd_0.csv'  
test_data_path = r'data\inflammatory_peptides_data\dd.csv'  


train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

print(f"Training data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")


print("Extracting ESM features...")
X_train_esm = []
for seq in train_data['sequence']:
    try:
        features = extract_esm_features(seq)
        X_train_esm.append(features)
    except Exception as e:
        print(f"Error processing sequence in training data: {e}")
        X_train_esm.append(np.zeros(2560))  

X_test_esm = []
for seq in test_data['sequence']:
    try:
        features = extract_esm_features(seq)
        X_test_esm.append(features)
    except Exception as e:
        print(f"Error processing sequence in test data: {e}")
        X_test_esm.append(np.zeros(2560))

X_train_esm = np.array(X_train_esm)
X_test_esm = np.array(X_test_esm)

print(f"ESM features - Train: {X_train_esm.shape}, Test: {X_test_esm.shape}")


print("Extracting CKSAAP features...")
train_sequences = [(f"seq_{i}", seq) for i, seq in enumerate(train_data['sequence'])]
test_sequences = [(f"seq_{i}", seq) for i, seq in enumerate(test_data['sequence'])]

X_train_cksaap = extract_cksaap_features(train_sequences, gap=5)
X_test_cksaap = extract_cksaap_features(test_sequences, gap=5)

print(f"CKSAAP features - Train: {X_train_cksaap.shape}, Test: {X_test_cksaap.shape}")


if X_train_esm.shape[0] != X_train_cksaap.shape[0]:
    min_samples = min(X_train_esm.shape[0], X_train_cksaap.shape[0])
    X_train_esm = X_train_esm[:min_samples]
    X_train_cksaap = X_train_cksaap[:min_samples]

if X_test_esm.shape[0] != X_test_cksaap.shape[0]:
    min_samples = min(X_test_esm.shape[0], X_test_cksaap.shape[0])
    X_test_esm = X_test_esm[:min_samples]
    X_test_cksaap = X_test_cksaap[:min_samples]


print("Combining features...")
X_train_combined = np.hstack((X_train_esm, X_train_cksaap))
X_test_combined = np.hstack((X_test_esm, X_test_cksaap))

print(f"Combined features - Train: {X_train_combined.shape}, Test: {X_test_combined.shape}")


print("Saving features...")
np.save('cksaap-features.npy', X_train_cksaap)
np.save('cksaap-test-features.npy', X_test_cksaap)
np.save('stage-2-features_3.npy', X_train_combined)
np.save('stage-2-test-features_3.npy', X_test_combined)

print("Feature extraction and saving complete.")