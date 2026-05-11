"""
features.py
Generación de variables predictoras (feature engineering) sobre datos OHLCV.
"""

import pandas as pd
import numpy as np


FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "return_1h", "return_3h", "return_6h",
    "range", "body",
    "ema_10", "ema_20",
    "volatility_10",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega variables derivadas al dataframe:
    - Retornos porcentuales (1h, 3h, 6h).
    - Rango y cuerpo de la vela.
    - EMAs de 10 y 20 períodos.
    - Volatilidad rolling de 10 períodos.
    Elimina las filas con NaN generadas por los cálculos rolling.
    """
    # Retornos
    df["return_1h"] = df["Close"].pct_change()
    df["return_3h"] = df["Close"].pct_change(3)
    df["return_6h"] = df["Close"].pct_change(6)

    # Morfología de la vela
    df["range"] = df["High"] - df["Low"]
    df["body"] = abs(df["Close"] - df["Open"])

    # Medias móviles exponenciales
    df["ema_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["ema_20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # Volatilidad
    df["volatility_10"] = df["return_1h"].rolling(window=10).std()

    df = df.dropna()
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Define la variable objetivo binaria:
    - 1: el precio de cierre de la próxima vela es mayor al actual.
    - 0: el precio no sube o baja.
    Elimina la última fila que queda con NaN tras el shift.
    """
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna()
    return df


def get_X_y(df: pd.DataFrame):
    """
    Separa features (X) y target (y) usando la lista FEATURES definida en el módulo.
    """
    X = df[FEATURES]
    y = df["target"]
    return X, y

