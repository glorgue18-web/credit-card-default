import pandas as pd
from pathlib import Path

def load_credit_data():
    file_path = Path(r"C:\Users\Gloria\Documents\Proyectos Personales\credit-card-default\data\raw\default of credit card clients.xls")
    df = pd.read_excel(file_path, header=1)
    return df
