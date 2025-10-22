import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import esm
import re
import math


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
    
    avg_pool = token_representations.mean(dim=1)
    max_pool = token_representations.max(dim=1).values
    
    pooled_output = torch.cat((avg_pool, max_pool), dim=1).squeeze().cpu().numpy()
    return pooled_output


def DDE(fastas, **kw):
    AA = kw.get('order', 'ACDEFGHIKLMNPQRSTVWY')

    myCodons = {
        'A': 4,
        'C': 2,
        'D': 2,
        'E': 2,
        'F': 2,
        'G': 4,
        'H': 2,
        'I': 3,
        'K': 2,
        'L': 6,
        'M': 1,
        'N': 2,
        'P': 4,
        'Q': 2,
        'R': 6,
        'S': 6,
        'T': 4,
        'V': 4,
        'W': 1,
        'Y': 2
    }

    encodings = []
    diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    header = ['#'] + diPeptides
    encodings.append(header)

    myTM = []
    for pair in diPeptides:
        myTM.append((myCodons[pair[0]] / 61) * (myCodons[pair[1]] / 61))

    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = [name]
        tmpCode = [0] * 400
        for j in range(len(sequence) - 1):
            tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j + 1]]] += 1
        if sum(tmpCode) != 0:
            tmpCode = [i / sum(tmpCode) for i in tmpCode]

        myTV = []
        for j in range(len(myTM)):
            myTV.append(myTM[j] * (1 - myTM[j]) / (len(sequence) - 1))

        for j in range(len(tmpCode)):
            tmpCode[j] = (tmpCode[j] - myTM[j]) / math.sqrt(myTV[j])

        code = code + tmpCode
        encodings.append(code)

   
    return [row[1:] for row in encodings[1:]]


train_data_path = r'data\inflammatory_peptides_data\dd_0.csv'  
test_data_path = r'data\inflammatory_peptides_data\dd.csv'  


train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)


X_train_esm = train_data['sequence'].apply(extract_esm_features).tolist()
X_test_esm = test_data['sequence'].apply(extract_esm_features).tolist()


train_sequences = [(f"seq_{i}", seq) for i, seq in enumerate(train_data['sequence'])]
test_sequences = [(f"seq_{i}", seq) for i, seq in enumerate(test_data['sequence'])]

X_train_dde = DDE(train_sequences)
X_test_dde = DDE(test_sequences)


X_train_combined = np.hstack((X_train_esm, X_train_dde))
X_test_combined = np.hstack((X_test_esm, X_test_dde))

print(X_train_combined.shape)
print(X_test_combined.shape)
np.save('esm-features.npy', X_train_esm)
np.save('esm-test-features.npy', X_test_esm)
np.save('dde-features.npy', X_train_dde)
np.save('dde-test-features.npy', X_test_dde)
np.save('stage-2-features_2.npy', X_train_combined)
np.save('stage-2-test-features_2.npy', X_test_combined)

print("Feature extraction and saving complete.")