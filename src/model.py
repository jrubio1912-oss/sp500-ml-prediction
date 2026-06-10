import numpy as np
import pandas as pd
import streamlit as st
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from src.features import FEATURES


DEFAULT_PARAMS = {
    'n_estimators':      500,
    'max_depth':         4,
    'num_leaves':        15,
    'learning_rate':     0.01,
    'min_child_samples': 40,
    'subsample':         0.8,
    'colsample_bytree':  0.7,
    'reg_alpha':         0.5,
    'reg_lambda':        5.0,
}


@st.cache_resource(show_spinner=False)
def train_model(_df: pd.DataFrame, train_end: str, params: dict):
    train_mask = _df.index < train_end
    X_train    = _df.loc[train_mask, FEATURES].dropna()
    y_train    = _df.loc[X_train.index, 'target']

    model = LGBMClassifier(
        **params,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model, X_train, y_train


def run_backtest(df: pd.DataFrame, model, test_start: str) -> tuple:
    test_mask = df.index >= test_start
    X_test    = df.loc[test_mask, FEATURES].dropna()
    y_test    = df.loc[X_test.index, 'target']

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    bt = df.loc[X_test.index, ['Close']].copy()
    bt['pred']      = preds
    bt['prob']      = proba
    bt['target']    = y_test
    bt['ret']       = bt['Close'].pct_change(1).shift(-1)
    bt['strat_ret'] = bt['pred'] * bt['ret']
    bt['bh_ret']    = bt['ret']
    bt = bt.dropna(subset=['strat_ret', 'bh_ret'])

    bt['strat_cum'] = (1 + bt['strat_ret']).cumprod()
    bt['bh_cum']    = (1 + bt['bh_ret']).cumprod()

    # Métricas solo con filas que tienen target real
    bt_eval = bt.dropna(subset=['target'])
    acc    = accuracy_score(bt_eval['target'], bt_eval['pred'])
    auc    = roc_auc_score(bt_eval['target'], bt_eval['prob'])
    
    sharpe = (bt['strat_ret'].mean() / bt['strat_ret'].std()) * np.sqrt(252)

    cum    = bt['strat_cum']
    dd     = (cum - cum.cummax()) / cum.cummax()
    max_dd = dd.min()

    metrics = {
        'accuracy':  acc,
        'auc':       auc,
        'sharpe':    sharpe,
        'max_dd':    max_dd,
        'ret_total': bt['strat_cum'].iloc[-1] - 1,
        'bh_total':  bt['bh_cum'].iloc[-1] - 1,
        'n_trades':  int(bt['pred'].sum()),
        'n_days':    len(bt),
    }
    return bt, metrics


def predict_next(df_full: pd.DataFrame, model) -> tuple:
    last      = df_full[FEATURES].dropna().iloc[[-1]]
    pred      = model.predict(last)[0]
    prob      = model.predict_proba(last)[0, 1]
    last_date = df_full.index[-1].date()
    return pred, prob, last_date