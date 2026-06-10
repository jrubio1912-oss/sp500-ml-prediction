import numpy as np
import pandas as pd


FEATURES = [
    'return_1d', 'return_3d', 'return_5d', 'return_10d', 'return_21d',
    'volatility_5d', 'volatility_21d', 'volatility_63d', 'vol_ratio',
    'volume_norm_5d', 'volume_norm_21d',
    'dist_ema5', 'dist_ema21', 'dist_ema63', 'dist_ema200',
    'rsi_14', 'macd', 'macd_signal', 'macd_hist',
    'bb_width', 'bb_position', 'day_of_week', 'month',
    'vix_level', 'vix_return_1d', 'vix_return_5d', 'vix_ma_ratio', 'vix_high_flag',
    'yield_level', 'yield_change_1d', 'yield_change_5d', 'yield_slope',
    'dxy_return_1d', 'dxy_return_5d',
    'gold_return_1d', 'gold_return_5d',
    'sp_vix_corr_21d', 'sp_gold_corr_21d',
]

FEATURE_GROUPS = {
    "Retornos":       ['return_1d','return_3d','return_5d','return_10d','return_21d'],
    "Volatilidad":    ['volatility_5d','volatility_21d','volatility_63d','vol_ratio'],
    "Volumen":        ['volume_norm_5d','volume_norm_21d'],
    "Tendencia":      ['dist_ema5','dist_ema21','dist_ema63','dist_ema200'],
    "Momentum":       ['rsi_14','macd','macd_signal','macd_hist','bb_width','bb_position'],
    "Calendario":     ['day_of_week','month'],
    "VIX":            ['vix_level','vix_return_1d','vix_return_5d','vix_ma_ratio','vix_high_flag'],
    "Tasas":          ['yield_level','yield_change_1d','yield_change_5d','yield_slope'],
    "DXY":            ['dxy_return_1d','dxy_return_5d'],
    "Oro":            ['gold_return_1d','gold_return_5d'],
    "Correlaciones":  ['sp_vix_corr_21d','sp_gold_corr_21d'],
}

UMBRAL = 0.002


def build_features(raw: dict) -> pd.DataFrame:
    sp500  = raw["sp500"]
    vix    = raw["vix"]
    yields = raw["yields"]
    dxy    = raw["dxy"]
    gold   = raw["gold"]

    df = sp500[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    # Retornos
    for n in [1, 3, 5, 10, 21]:
        df[f'return_{n}d'] = df['Close'].pct_change(n)

    # Volatilidad
    for n in [5, 21, 63]:
        df[f'volatility_{n}d'] = df['return_1d'].rolling(n).std()
    df['vol_ratio'] = df['volatility_5d'] / df['volatility_21d']

    # Volumen
    df['volume_norm_5d']  = df['Volume'] / df['Volume'].rolling(5).mean()
    df['volume_norm_21d'] = df['Volume'] / df['Volume'].rolling(21).mean()

    # EMAs
    for span in [5, 21, 63, 200]:
        df[f'dist_ema{span}'] = (
            (df['Close'] - df['Close'].ewm(span=span).mean()) / df['Close']
        )

    # RSI
    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df['rsi_14'] = 100 - (100 / (1 + gain / loss))

    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['macd']        = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']

    # Bollinger
    bb_sma = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    bb_u   = bb_sma + 2 * bb_std
    bb_l   = bb_sma - 2 * bb_std
    df['bb_width']    = (bb_u - bb_l) / df['Close']
    df['bb_position'] = (df['Close'] - bb_l) / (bb_u - bb_l)

    # Calendario
    df['day_of_week'] = df.index.dayofweek
    df['month']       = df.index.month

    # Externos
    def align(s):
        return s.reindex(df.index, method='ffill')

    v = align(vix['Close'])
    df['vix_level']     = v
    df['vix_return_1d'] = v.pct_change(1)
    df['vix_return_5d'] = v.pct_change(5)
    df['vix_ma_ratio']  = v / v.rolling(21).mean()
    df['vix_high_flag'] = (v > 30).astype(int)

    y10 = align(yields['Close'])
    df['yield_level']     = y10
    df['yield_change_1d'] = y10.pct_change(1)
    df['yield_change_5d'] = y10.pct_change(5)
    df['yield_slope']     = y10 - y10.rolling(21).mean()

    d = align(dxy['Close'])
    df['dxy_return_1d'] = d.pct_change(1)
    df['dxy_return_5d'] = d.pct_change(5)

    g = align(gold['Close'])
    df['gold_return_1d'] = g.pct_change(1)
    df['gold_return_5d'] = g.pct_change(5)

    df['sp_vix_corr_21d']  = df['return_1d'].rolling(21).corr(df['vix_return_1d'])
    df['sp_gold_corr_21d'] = df['return_1d'].rolling(21).corr(df['gold_return_1d'])

    # Target
    ret_fut    = df['Close'].pct_change(1).shift(-1)
    df['target'] = np.where(
        ret_fut >  UMBRAL, 1,
        np.where(ret_fut < -UMBRAL, 0, np.nan)
    )
    # No dropeamos la última fila — se usa para predecir el día siguiente
    df_model = df.dropna(subset=['target']).copy()
    df_model['target'] = df_model['target'].astype(int)

    return df, df_model
    