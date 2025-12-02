import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    # 1. Eliminar ID (no aporta a predicción)
    df = df.drop(columns=["ID"])

    # 2. Renombrar target
    df = df.rename(columns={"default payment next month": "default"})

    # 3. Separar X / y
    X = df.drop(columns=["default"])
    y = df["default"]

    # 4. Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Escalador para modelos lineales
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
