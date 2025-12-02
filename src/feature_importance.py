import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

def plot_feature_importance(model, feature_names, top_n=20):
    """
    Genera un gráfico profesional de importancia de variables usando el GAIN.
    """
    importance = model.get_booster().get_score(importance_type='gain')
    
    # Convertir a un DataFrame
    imp_df = pd.DataFrame({
        "feature": list(importance.keys()),
        "importance_gain": list(importance.values())
    }).sort_values("importance_gain", ascending=False)

    print("\n===== IMPORTANCIA DE VARIABLES (GAIN) =====")
    print(imp_df.head(top_n))

    # Gráfico
    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=imp_df.head(top_n),
        y="feature",
        x="importance_gain",
        palette="viridis"
    )
    plt.title("Top {} Variables más importantes (XGBoost - GAIN)".format(top_n))
    plt.xlabel("Importancia (Gain)")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.show()

    return imp_df
