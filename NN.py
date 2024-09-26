# Train Random Forest
from sklearn.neural_network import MLPClassifier

# Define the MLPClassifier with 4 hidden layers, each having half the neurons of the previous layer
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50, 50, 50),  # 4 hidden layers with decreasing neurons
    activation='tanh',
    alpha=0.0070,
    learning_rate='invscaling',
    solver='sgd',
    random_state=42
)
mlp.fit(np.array(X_train), y_train)

y_pred_train_prob = mlp.predict_proba(np.array(X_train))[:, 1]
y_pred_train= mlp.predict(np.array(X_train))     


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


y_pred_test_prob = mlp.predict_proba(np.array(X_test))[:, 1]
y_pred_test= mlp.predict(np.array(X_test))     


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
