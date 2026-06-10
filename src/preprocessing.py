import pandas as pd
import yfinance as yf
import streamlit as st


TICKERS = {
    "sp500":  "^GSPC",
    "vix":    "^VIX",
    "yields": "^TNX",
    "dxy":    "DX-Y.NYB",
    "gold":   "GC=F",
}


def download_data(start: str, end: str) -> dict:
    raw = {}
    for name, ticker in TICKERS.items():
        df = yf.download(ticker, start=start, end=end,
                         interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        raw[name] = df
    return raw