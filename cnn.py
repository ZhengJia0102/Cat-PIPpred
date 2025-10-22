import random
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.regularizers import l2
from keras.backend import sigmoid
import os
import math
from keras.utils import get_custom_objects
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, matthews_corrcoef, average_precision_score, confusion_matrix
from keras.layers import *
from keras.models import *
from keras import backend as K


random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'


def swish(x, beta=1):
    return (x * sigmoid(beta * x))

get_custom_objects().update({'swish': swish})


def normalization_layer(input_layer):
    output_layer = Lambda(lambda x: x / K.max(x, axis=-1, keepdims=True))(input_layer)
    return output_layer


def calculate_MCC(TP, TN, FP, FN):
    NUMERATOR = (TP * TN) - (FP * FN)
    DENOMINATOR = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    
    if DENOMINATOR == 0:
        MCC = 0  
    else:
        MCC = NUMERATOR / DENOMINATOR
    
    return MCC


def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.8
    epochs_drop = 5.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


def createModel1():
    word_input = Input((27, 193), name='word_input')
    
    # Path 1
    overallResult = Convolution1D(filters=128, kernel_size=3, padding='same', activation="relu")(word_input)
    overallResult = MaxPooling1D(pool_size=2)(overallResult)
    flatten = Flatten()(overallResult)
    
    # Path 2
    overallResult1 = Convolution1D(filters=128, kernel_size=5, padding='same', activation='relu')(word_input)
    overallResult1 = MaxPooling1D(pool_size=2)(overallResult1)
    overallResult1 = GlobalMaxPooling1D()(overallResult1)
    
    merged = Concatenate()([normalization_layer(overallResult1), normalization_layer(flatten)])
    
    dense1 = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(merged)
    dense2 = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(dense1)
    ss_output = Dense(1, activation='sigmoid', name='ss_output')(dense2)
    
    return Model(inputs=[word_input], outputs=[ss_output])


X_train = np.load('combine_feature_80.npy', allow_pickle=True)
print("Shape of X_train:", X_train.shape)
train_data = pd.read_excel(r'data\inflammatory_peptides_data\dd_0.xlsx')  
y_train = train_data['label'].values


X_train = np.reshape(X_train, (X_train.shape[0], 27, 193))


print("Reshaped X_train:", X_train.shape)


n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


fold_results = []

for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
    print(f"Training fold {fold + 1}/{n_splits}")
    
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    
    
    model = createModel1()
    model.compile(optimizer='adam', loss={'ss_output': 'binary_crossentropy'}, metrics=['accuracy'])
    
    
    checkpoint = ModelCheckpoint(f"model_fold_{fold + 1}.h5", monitor='val_accuracy', verbose=0, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=20, verbose=0, mode='auto')
    lr_scheduler = LearningRateScheduler(step_decay)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
    
    
    history = model.fit(
        {'word_input': X_train_fold},
        {'ss_output': y_train_fold},
        epochs=20,
        batch_size=32,
        callbacks=[early_stopping, checkpoint, lr_scheduler, reduce_lr],
        verbose=2,
        validation_data=({'word_input': X_val_fold}, {'ss_output': y_val_fold}),
        shuffle=True
    )
    
    
    model.load_weights(f"model_fold_{fold + 1}.h5")
    
    
    y_pred = model.predict({'word_input': X_val_fold})
    y_pred_proba = y_pred[:, 0]  
    y_pred_class = (y_pred > 0.5).astype(int)  
    
    
    accuracy = accuracy_score(y_val_fold, y_pred_class)
    roc_auc = roc_auc_score(y_val_fold, y_pred_proba)
    mcc = matthews_corrcoef(y_val_fold, y_pred_class)
    avg_precision = average_precision_score(y_val_fold, y_pred_proba)
    
    
    cm = confusion_matrix(y_val_fold, y_pred_class)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity (SN)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity (SP)
    
    print(f"Fold {fold + 1} results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Matthews Correlation Coefficient: {mcc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Sensitivity (SN): {sensitivity:.4f}")
    print(f"Specificity (SP): {specificity:.4f}")
    
    fold_results.append({
        'fold': fold + 1,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'mcc': mcc,
        'avg_precision': avg_precision,
        'sensitivity': sensitivity,
        'specificity': specificity
    })


print("\nCross-validation results for each fold:")
for result in fold_results:
    print(result)


mean_results = {
    'accuracy': np.mean([result['accuracy'] for result in fold_results]),
    'roc_auc': np.mean([result['roc_auc'] for result in fold_results]),
    'mcc': np.mean([result['mcc'] for result in fold_results]),
    'avg_precision': np.mean([result['avg_precision'] for result in fold_results]),
    'sensitivity': np.mean([result['sensitivity'] for result in fold_results]),
    'specificity': np.mean([result['specificity'] for result in fold_results])
}

print("\nMean results over 5 folds:")
print(f"Accuracy: {mean_results['accuracy']:.4f}")
print(f"ROC AUC: {mean_results['roc_auc']:.4f}")
print(f"Matthews Correlation Coefficient: {mean_results['mcc']:.4f}")
print(f"Average Precision: {mean_results['avg_precision']:.4f}")
print(f"Sensitivity (SN): {mean_results['sensitivity']:.4f}")
print(f"Specificity (SP): {mean_results['specificity']:.4f}")