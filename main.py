import matplotlib
matplotlib.use('TkAgg')

from src.load_data import load_credit_data
from src.preprocess import preprocess_data
from src.model_xgb import run_xgboost
from src.feature_importance import plot_feature_importance
from src.evaluation import plot_confusion_matrix, plot_roc_curve, plot_precision_recall
from src.threshold import find_best_threshold
from src.shap_analysis import shap_global_summary, shap_explain_single

def main():
    # 1. Cargar datos
    df = load_credit_data()

    # 2. Preprocesamiento
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = preprocess_data(df)

    # 3. Entrenamiento del modelo XGBoost
    model, preds, probs = run_xgboost(X_train, X_test, y_train, y_test)

    # 4. Importancia de variables
    plot_feature_importance(model, X_train.columns)

    # 5. Matriz de confusión
    plot_confusion_matrix(y_test, preds)

    # 6. Curva ROC
    plot_roc_curve(y_test, probs)

    # 7. Curva Precision–Recall
    plot_precision_recall(y_test, probs)

    # 8. Optimización de umbrales
    thresholds = find_best_threshold(y_test, probs)
    print("\n===== OPTIMIZACIÓN DE UMBRALES =====")
    for k, v in thresholds.items():
        print(f"{k}: {v}")

    # 9. SHAP: explicabilidad global
    shap_global_summary(model, X_train)

    print("\n===== SHAP — EXPLICACIÓN LOCAL PARA UN CLIENTE =====")
    shap_explain_single(model, X_test, index=0, top_n=10)

if __name__ == "__main__":
    main()
