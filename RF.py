from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score,
    recall_score, matthews_corrcoef, roc_curve, precision_recall_curve
)
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# Train Random Forest
rf_model = RandomForestClassifier(max_depth=3, max_features='log2', min_samples_leaf=1, min_samples_split=8, n_estimators=270, n_jobs=40, random_state=42)
rf_model.fit(np.array(X_train), y_train)

y_pred_train_prob = rf_model.predict_proba(np.array(X_train))[:, 1]
y_pred_train= rf_model.predict(np.array(X_train))     


# Calculate metrics
avg_precision = average_precision_score(y_train, y_pred_train_prob)
roc_auc = roc_auc_score(y_train, y_pred_train_prob)
precision = precision_score(y_train, y_pred_train)
recall = recall_score(y_train, y_pred_train)
mcc = matthews_corrcoef(y_train, y_pred_train)
f1 = f1_score(y_train, y_pred_train)
print('TRAIN SET')
print(f'PR-AUC: {avg_precision:.4f}')
print(f'ROC-AUC: {roc_auc:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'MCC: {mcc:.4f}')

y_pred_test_prob = rf_model.predict_proba(np.array(X_test))[:, 1]
y_pred_test= rf_model.predict(np.array(X_test))     


# Calculate metrics
avg_precision = average_precision_score(y_test, y_pred_test_prob)
roc_auc = roc_auc_score(y_test, y_pred_test_prob)
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
mcc = matthews_corrcoef(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)
print('TEST SET')
print(f'PR-AUC: {avg_precision:.4f}')
print(f'ROC-AUC: {roc_auc:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'MCC: {mcc:.4f}')
