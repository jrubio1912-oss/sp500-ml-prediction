"""
preprocessing.py
Carga, limpieza y preparación del dataset histórico del S&P 500.
"""

import pandas as pd
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    """
    Carga el CSV del S&P 500 omitiendo las primeras filas de metadata
    que genera Yahoo Finance (filas 1 y 2).
    """
    df = pd.read_csv(filepath, skiprows=[1, 2])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y prepara el dataframe:
    - Renombra la columna de fecha si viene como 'Price'.
    - Convierte la columna Datetime al tipo correcto.
    - Ordena cronológicamente y establece Datetime como índice.
    - Convierte las columnas OHLCV a tipo numérico.
    - Elimina filas con valores nulos.
    """
    # Renombrar columna de fecha si es necesario
    if "Price" in df.columns:
        df = df.rename(columns={"Price": "Datetime"})

    # Recuperar Datetime desde índice si es necesario
    if "Datetime" not in df.columns and df.index.name == "Datetime":
        df = df.reset_index()

    # Limpiar y parsear fechas
    df = df.dropna(subset=["Datetime"])
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    # Ordenar y setear índice temporal
    df = df.sort_values("Datetime")
    df.set_index("Datetime", inplace=True)

    # Convertir columnas numéricas
    cols_numericas = ["Open", "High", "Low", "Close", "Volume"]
    for col in cols_numericas:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Eliminar nulos
    df = df.dropna()

    return df


def split_temporal(X: pd.DataFrame, y: pd.Series, train_ratio: float = 0.8):
    """
    Divide los datos respetando el orden temporal (sin shuffle).
    Retorna X_train, X_test, y_train, y_test.
    """
    split_index = int(len(X) * train_ratio)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print(f"Train: {len(X_train):,} muestras  ({X_train.index[0]} → {X_train.index[-1]})")
    print(f"Test : {len(X_test):,}  muestras  ({X_test.index[0]} → {X_test.index[-1]})")

    return X_train, X_test, y_train, y_test

