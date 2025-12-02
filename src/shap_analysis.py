import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# shap.initjs()

# ============================
# EXPLICABILIDAD GLOBAL
# ============================
def shap_global_summary(model, X_train, max_samples=2000):
    """
    Crea un summary plot global de SHAP para entender
    qué variables empujan más el riesgo en toda la cartera.
    """
    # Muestrita para que no tarde siglos
    if len(X_train) > max_samples:
        X_sample = X_train.sample(max_samples, random_state=42)
    else:
        X_sample = X_train.copy()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Summary plot
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=X_sample.columns,
        show=True
    )


# ============================
# EXPLICABILIDAD LOCAL
# ============================
def shap_explain_single(model, X_test, index=0, top_n=10):
    """
    Explicación local: muestra las variables que más empujan
    la predicción de riesgo para un cliente en particular.
    """
    explainer = shap.TreeExplainer(model)

    # Tomamos un cliente específico del set de test
    x_row = X_test.iloc[[index]]  # mantiene DataFrame

    shap_values = explainer.shap_values(x_row)
    base_value = explainer.expected_value

    # A veces shap_values es lista (por clase). Si es así, tomamos la clase 1.
    if isinstance(shap_values, list):
        shap_row = shap_values[1][0]
        base = base_value[1]
    else:
        shap_row = shap_values[0]
        base = base_value

    # Reconstruimos la log-odds y la probabilidad
    log_odds = base + shap_row.sum()
    proba = 1 / (1 + np.exp(-log_odds))

    contrib_df = pd.DataFrame({
        "feature": X_test.columns,
        "shap_value": shap_row,
        "abs_shap": np.abs(shap_row),
        "feature_value": x_row.values[0]
    }).sort_values("abs_shap", ascending=False)

    print("\n===== EXPLICACIÓN LOCAL (SHAP) PARA UN CLIENTE =====")
    print(f"Índice en X_test: {index}")
    print(f"Probabilidad estimada de default: {proba:.4f}\n")

    print(f"Top {top_n} variables que más influyen en la predicción:\n")
    print(contrib_df.head(top_n)[["feature", "feature_value", "shap_value", "abs_shap"]])

    # Opcional: waterfall plot si quieres una gráfica local
    try:
        shap.plots._waterfall.waterfall_legacy(
            base,
            shap_row,
            feature_names=X_test.columns
        )
        plt.show()
    except Exception as e:
        print("\n(No se pudo generar waterfall plot, pero la explicación tabular está arriba.)")
        print("Error:", e)
