from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

def run_logistic(X_train_scaled, X_test_scaled, y_train, y_test):
    model = LogisticRegression(class_weight="balanced", max_iter=200)
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    proba = model.predict_proba(X_test_scaled)[:, 1]

    print("\n===== REGRESIÓN LOGÍSTICA =====")
    print(classification_report(y_test, preds))
    print("ROC AUC:", roc_auc_score(y_test, proba))
