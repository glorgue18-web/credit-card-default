import matplotlib
matplotlib.use('TkAgg')

from src.load_data import load_credit_data
from src.preprocess import preprocess_data
from src.model_xgb import run_xgboost
from src.feature_importance import plot_feature_importance
from src.threshold import find_best_threshold
from src.evaluation import plot_confusion_matrix, plot_roc_curve, plot_precision_recall
from src.shap_analysis import shap_global_summary, shap_explain_single

def main():

    print("\n===== CARGANDO DATOS =====")
    df = load_credit_data()

    print("\n===== PREPROCESAMIENTO =====")
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = preprocess_data(df)

    print("\n===== ENTRENANDO MODELO XGBOOST =====")
    model, preds, probs = run_xgboost(X_train, X_test, y_train, y_test)

    print("\n===== IMPORTANCIA DE VARIABLES =====")
    plot_feature_importance(model, X_train.columns)

    print("\n===== MATRIZ DE CONFUSIÓN =====")
    plot_confusion_matrix(y_test, preds)

    print("\n===== CURVA ROC =====")
    plot_roc_curve(y_test, probs)

    print("\n===== CURVA PRECISION-RECALL =====")
    plot_precision_recall(y_test, probs)

    print("\n===== OPTIMIZACIÓN DE UMBRALES =====")
    thresholds = find_best_threshold(y_test, probs)
    for k, v in thresholds.items():
        print(f"{k}: {v}")

    print("\n===== SHAP — EXPLICABILIDAD GLOBAL =====")
    shap_global_summary(model, X_train)

    print("\n===== SHAP — EXPLICACIÓN LOCAL (CLIENTE 0) =====")
    shap_explain_single(model, X_test, index=0, top_n=10)


if __name__ == "__main__":
    main()
