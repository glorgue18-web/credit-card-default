from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

def run_random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    print("\n===== RANDOM FOREST =====")
    print(classification_report(y_test, preds))
    print("ROC AUC:", roc_auc_score(y_test, proba))
