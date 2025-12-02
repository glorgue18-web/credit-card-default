import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# =======================
# 1. INFO GENERAL
# =======================
def general_info(df):
    print("\n===== SHAPE =====")
    print(df.shape)

    print("\n===== COLUMNAS =====")
    print(list(df.columns))

    print("\n===== TIPOS DE DATOS =====")
    print(df.dtypes)

    print("\n===== PRIMERAS FILAS =====")
    print(df.head())

    print("\n===== RESUMEN ESTADÍSTICO =====")
    print(df.describe())


# =======================
# 2. NULOS
# =======================
def missing_values(df):
    print("\n===== VALORES NULOS =====")
    print(df.isnull().sum())


# =======================
# 3. BALANCE DE CLASES
# =======================
def class_balance(df, target_col="default payment next month"):
    print("\n===== BALANCE DE CLASES =====")
    print(df[target_col].value_counts())

    df[target_col].value_counts().plot(kind="bar")
    plt.title("Distribución de la variable objetivo (Default)")
    plt.xlabel("Clase (0=No Default, 1=Default)")
    plt.ylabel("Frecuencia")
    plt.show()


# =======================
# 4. EDAD, SEXO, EDUCACIÓN, MATRIMONIO
# =======================
def demographics_analysis(df):
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    sns.countplot(data=df, x="SEX", ax=ax[0][0])
    ax[0][0].set_title("Distribución por Sexo")

    sns.countplot(data=df, x="EDUCATION", ax=ax[0][1])
    ax[0][1].set_title("Distribución por Nivel Educativo")

    sns.countplot(data=df, x="MARRIAGE", ax=ax[1][0])
    ax[1][0].set_title("Distribución por Estado Civil")

    sns.histplot(df["AGE"], bins=20, kde=True, ax=ax[1][1])
    ax[1][1].set_title("Distribución de Edad")

    plt.tight_layout()
    plt.show()


# =======================
# 5. ESTATUS DE PAGO PAY_0 ... PAY_6
# =======================
def payments_status(df):
    pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

    df[pay_cols].hist(bins=15, figsize=(12, 8))
    plt.suptitle("Distribución del Estatus de Pago")
    plt.show()


# =======================
# 6. VARIABLES FINANCIERAS BILL_AMT y PAY_AMT
# =======================
def financial_variables(df):
    bill_cols = [c for c in df.columns if "BILL_AMT" in c]
    pay_cols = [c for c in df.columns if "PAY_AMT" in c]

    df[bill_cols].hist(figsize=(14, 8), bins=20)
    plt.suptitle("Distribución de montos de facturación (BILL_AMT)")
    plt.show()

    df[pay_cols].hist(figsize=(14, 8), bins=20)
    plt.suptitle("Distribución de montos de pago (PAY_AMT)")
    plt.show()


# =======================
# 7. CORRELACIONES
# =======================
def correlation_heatmap(df):
    plt.figure(figsize=(16, 12))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
    plt.title("Mapa de calor de correlaciones")
    plt.show()


# =======================
# FUNCIÓN MAESTRA PARA EL EDA COMPLETO
# =======================
def full_eda(df):
    general_info(df)
    missing_values(df)
    class_balance(df)
    demographics_analysis(df)
    payments_status(df)
    financial_variables(df)
    correlation_heatmap(df)
