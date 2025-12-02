import matplotlib
matplotlib.use('TkAgg')

from src.load_data import load_credit_data
from src.regression_pay_amt4 import run_regression_models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def main():

    df = load_credit_data()

    # Variables anteriores a abril
    features = [
        "SEX", "EDUCATION", "MARRIAGE", "AGE", "LIMIT_BAL",
        "PAY_0", "PAY_2", "PAY_3",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3"
    ]

    X = df[features]
    y = df["PAY_AMT4"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = run_regression_models(
        pd.DataFrame(X_train_scaled, columns=features),
        pd.DataFrame(X_test_scaled, columns=features),
        y_train, y_test
    )

    print("\n===== RESULTADOS MODELOS DE REGRESIÓN (PAY_AMT4) =====")
    for name, r in results.items():
        print(f"\n→ {name}")
        print(f"MAE:  {r['MAE']:.4f}")
        print(f"RMSE: {r['RMSE']:.4f}")
        print(f"R²:   {r['R2']:.4f}")

if __name__ == "__main__":
    main()
