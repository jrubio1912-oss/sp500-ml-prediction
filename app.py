import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from src.preprocessing import download_data
from src.features import build_features, FEATURES, FEATURE_GROUPS
from src.model import train_model, run_backtest, predict_next

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SP500 · Direction Model",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# ESTILOS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0e1117;
    color: #e2e8f0;
  }
  .metric-card {
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 20px 24px;
    text-align: center;
  }
  .metric-label {
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #718096;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 6px;
  }
  .metric-value {
    font-size: 28px;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    color: #e2e8f0;
  }
  .metric-value.green { color: #48bb78; }
  .metric-value.red   { color: #fc8181; }
  .metric-value.amber { color: #f6ad55; }

  .signal-box {
    border-radius: 10px;
    padding: 24px 28px;
    text-align: center;
    margin-bottom: 12px;
  }
  .signal-up   { background: #1a2e22; border: 2px solid #48bb78; }
  .signal-down { background: #2d1b1b; border: 2px solid #fc8181; }
  .signal-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #718096;
    margin-bottom: 8px;
  }
  .signal-arrow { font-size: 48px; line-height: 1; }
  .signal-text {
    font-size: 20px;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 6px;
  }
  .signal-prob {
    font-size: 12px;
    color: #a0aec0;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 6px;
  }
  .section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4a5568;
    border-bottom: 1px solid #2d3748;
    padding-bottom: 8px;
    margin-bottom: 20px;
    margin-top: 32px;
  }
  div[data-testid="stSidebar"] {
    background-color: #13161f;
    border-right: 1px solid #1e2535;
  }
  .stButton > button {
    width: 100%;
    background: #2d3a8c;
    color: #e2e8f0;
    border: none;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    letter-spacing: 0.05em;
    padding: 10px;
  }
  .stButton > button:hover { background: #3b4db5; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
PLOTLY_DARK = dict(
    template="plotly_dark",
    paper_bgcolor='#0e1117',
    plot_bgcolor='#13161f',
    font=dict(color='#a0aec0', family='IBM Plex Mono'),
    margin=dict(l=0, r=0, t=10, b=0),
    xaxis=dict(gridcolor='#1e2535'),
    yaxis=dict(gridcolor='#1e2535'),
)

def metric_card(label, value, cls=""):
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value {cls}">{value}</div>
    </div>""", unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### SP500 · Model Config")
    st.markdown("---")

    train_years     = st.slider("Años de entrenamiento", 5, 20, 18, step=1)
    test_split_year = st.selectbox("Test desde", [2022, 2023, 2024], index=1)

    st.markdown("**Hiperparámetros LightGBM**")
    n_est      = st.select_slider("n_estimators",  [200, 300, 500, 800], value=500)
    max_depth  = st.select_slider("max_depth",      [3, 4, 5],            value=4)
    lr         = st.select_slider("learning_rate",  [0.005, 0.01, 0.02],  value=0.01)
    num_leaves = st.select_slider("num_leaves",      [7, 15, 31],          value=15)
    reg_lambda = st.select_slider("reg_lambda",      [1.0, 5.0, 10.0],     value=5.0)

    st.markdown("---")
    st.markdown("**Gráfico de precios**")
    chart_days = st.slider("Días a mostrar", 30, 500, 180, step=10)
    show_indicators = st.multiselect(
        "Indicadores",
        ["SMA 20", "SMA 50", "Bollinger Bands", "Volumen"],
        default=["SMA 20", "Bollinger Bands"],
    )

    st.markdown("---")
    run_button    = st.button("▶  Entrenar y evaluar")
    update_button = st.button("🔄  Actualizar datos")

    if update_button:
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Caché limpiado. Presioná ▶ para reentrenar.")

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:24px 0 8px; border-bottom:1px solid #2d3748; margin-bottom:8px;'>
  <span style='font-family:IBM Plex Mono,monospace; font-size:13px;
               letter-spacing:0.15em; text-transform:uppercase; color:#4a5568;'>
    SP500 · DIRECTION MODEL
  </span>
  <h1 style='margin:6px 0 4px; font-size:28px; font-weight:600; color:#e2e8f0;'>
    ¿Sube o baja el S&P500 mañana?
  </h1>
  <p style='color:#718096; font-size:14px; margin:0;'>
    LightGBM · 38 features técnicos y macroeconómicos · datos diarios
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────
if run_button:
    end_date = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=train_years * 365 + 730)).strftime("%Y-%m-%d")
    train_end  = f"{test_split_year}-01-01"

    params = {
        'n_estimators':      n_est,
        'max_depth':         max_depth,
        'learning_rate':     lr,
        'num_leaves':        num_leaves,
        'min_child_samples': 40,
        'subsample':         0.8,
        'colsample_bytree':  0.7,
        'reg_alpha':         0.5,
        'reg_lambda':        reg_lambda,
    }

    with st.spinner("Descargando datos del mercado..."):
        raw          = download_data(start_date, end_date)
        df_full, df  = build_features(raw)
        df           = df.dropna(subset=FEATURES + ['target'])

    with st.spinner("Entrenando modelo..."):
        model, X_train, y_train = train_model(df, train_end, params)

    with st.spinner("Evaluando..."):
        bt, metrics                     = run_backtest(df_full, model, train_end)
        pred_next, prob_next, last_date = predict_next(df_full, model)

    st.session_state.update({
        'ready': True, 'model': model, 'df': df_full,
        'bt': bt, 'metrics': metrics,
        'pred_next': pred_next, 'prob_next': prob_next,
        'last_date': last_date,
        'X_train': X_train, 'y_train': y_train,
    })

# ─────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────
if st.session_state.get('ready'):
    metrics   = st.session_state['metrics']
    bt        = st.session_state['bt']
    model     = st.session_state['model']
    df        = st.session_state['df']
    pred_next = st.session_state['pred_next']
    prob_next = st.session_state['prob_next']
    last_date = st.session_state['last_date']
    X_train   = st.session_state['X_train']
    y_train   = st.session_state['y_train']

    # ── 1. SEÑAL ─────────────────────────────────────────────
    section("SEÑAL · PRÓXIMO DÍA HÁBIL")

    col_sig, col_metrics = st.columns([1, 2])

    with col_sig:
        is_up  = pred_next == 1
        css    = "signal-up"   if is_up else "signal-down"
        arrow  = "↑"           if is_up else "↓"
        texto  = "SUBE"        if is_up else "BAJA"
        color  = "#48bb78"     if is_up else "#fc8181"
        conf   = "Alta"        if abs(prob_next - 0.5) > 0.1 else "Baja"

        st.markdown(f"""
        <div class="signal-box {css}">
          <div class="signal-label">Señal · {last_date}</div>
          <div class="signal-arrow" style="color:{color}">{arrow}</div>
          <div class="signal-text" style="color:{color}">{texto}</div>
          <div class="signal-prob">P(Sube) = {prob_next:.3f} · Confianza {conf}</div>
        </div>""", unsafe_allow_html=True)

        # Barra de probabilidad
        prob_baja = 1 - prob_next
        st.markdown(f"**↓ Baja** — {prob_baja:.1%}")
        st.progress(float(prob_baja))
        st.markdown(f"**↑ Sube** — {prob_next:.1%}")
        st.progress(float(prob_next))

    with col_metrics:
        r1c1, r1c2 = st.columns(2)
        r2c1, r2c2 = st.columns(2)
        with r1c1: metric_card("Accuracy",  f"{metrics['accuracy']:.3f}", "green" if metrics['accuracy'] > 0.52 else "amber")
        with r1c2: metric_card("AUC-ROC",   f"{metrics['auc']:.3f}",      "green" if metrics['auc']      > 0.53 else "amber")
        with r2c1: metric_card("Sharpe",    f"{metrics['sharpe']:.2f}",   "green" if metrics['sharpe']   > 0.5  else "amber")
        with r2c2: metric_card("Max DD",    f"{metrics['max_dd']*100:.1f}%", "red" if metrics['max_dd'] < -0.15 else "amber")

    # ── 2. GRÁFICO DE VELAS ──────────────────────────────────
    section("PRECIO · S&P 500")

    df_chart = df.tail(chart_days).copy()

    if "Volumen" in show_indicators:
        fig_price = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.75, 0.25], vertical_spacing=0.03,
        )
    else:
        fig_price = make_subplots(rows=1, cols=1)

    fig_price.add_trace(go.Candlestick(
        x=df_chart.index,
        open=df_chart['Open'], high=df_chart['High'],
        low=df_chart['Low'],   close=df_chart['Close'],
        name="S&P 500",
        increasing_line_color='#48bb78',
        decreasing_line_color='#fc8181',
    ), row=1, col=1)

    if "SMA 20" in show_indicators:
        fig_price.add_trace(go.Scatter(
            x=df_chart.index, y=df_chart['Close'].rolling(20).mean(),
            name="SMA 20", line=dict(color='#4da6ff', width=1.5),
        ), row=1, col=1)

    if "SMA 50" in show_indicators:
        fig_price.add_trace(go.Scatter(
            x=df_chart.index, y=df_chart['Close'].rolling(50).mean(),
            name="SMA 50", line=dict(color='#f6ad55', width=1.5),
        ), row=1, col=1)

    if "Bollinger Bands" in show_indicators:
        sma20 = df_chart['Close'].rolling(20).mean()
        std20 = df_chart['Close'].rolling(20).std()
        fig_price.add_trace(go.Scatter(
            x=df_chart.index, y=sma20 + 2*std20,
            name="BB Sup", line=dict(color='#9966ff', width=1, dash='dot'),
        ), row=1, col=1)
        fig_price.add_trace(go.Scatter(
            x=df_chart.index, y=sma20 - 2*std20,
            name="BB Inf", line=dict(color='#9966ff', width=1, dash='dot'),
            fill='tonexty', fillcolor='rgba(153,102,255,0.05)',
        ), row=1, col=1)

    if "Volumen" in show_indicators:
        vol_colors = ['#48bb78' if c >= o else '#fc8181'
                      for c, o in zip(df_chart['Close'], df_chart['Open'])]
        fig_price.add_trace(go.Bar(
            x=df_chart.index, y=df_chart['Volume'],
            name="Volumen", marker_color=vol_colors, opacity=0.6,
        ), row=2, col=1)

    fig_price.update_layout(
        **PLOTLY_DARK,
        height=480,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, bgcolor='#1a1f2e'),
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # ── 3. TABS INDICADORES ──────────────────────────────────
    section("INDICADORES TÉCNICOS")

    tab_rsi, tab_macd, tab_vix = st.tabs(["RSI", "MACD", "VIX"])

    df_ind = df.tail(chart_days)

    with tab_rsi:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=df_ind.index, y=df_ind['rsi_14'],
            name="RSI 14", line=dict(color='#4da6ff', width=2),
        ))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="#fc8181",  annotation_text="Sobrecompra (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="#48bb78",  annotation_text="Sobreventa (30)")
        fig_rsi.add_hline(y=50, line_dash="dot",  line_color="#4a5568")
        fig_rsi.update_layout(**PLOTLY_DARK, height=250)
        fig_rsi.update_yaxes(range=[0, 100], gridcolor='#1e2535')
        st.plotly_chart(fig_rsi, use_container_width=True)

        rsi_val = df_ind['rsi_14'].iloc[-1]
        if rsi_val > 70:
            st.warning(f"RSI en {rsi_val:.1f} — zona de sobrecompra.")
        elif rsi_val < 30:
            st.success(f"RSI en {rsi_val:.1f} — zona de sobreventa.")
        else:
            st.info(f"RSI en {rsi_val:.1f} — zona neutral.")

    with tab_macd:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(
            x=df_ind.index, y=df_ind['macd'],
            name="MACD", line=dict(color='#4da6ff', width=2),
        ))
        fig_macd.add_trace(go.Scatter(
            x=df_ind.index, y=df_ind['macd_signal'],
            name="Signal", line=dict(color='#f6ad55', width=2),
        ))
        hist_colors = ['#48bb78' if v >= 0 else '#fc8181' for v in df_ind['macd_hist']]
        fig_macd.add_trace(go.Bar(
            x=df_ind.index, y=df_ind['macd_hist'],
            name="Histograma", marker_color=hist_colors, opacity=0.7,
        ))
        fig_macd.update_layout(**PLOTLY_DARK, height=250)
        st.plotly_chart(fig_macd, use_container_width=True)

    with tab_vix:
        fig_vix = go.Figure()
        fig_vix.add_trace(go.Scatter(
            x=df_ind.index, y=df_ind['vix_level'],
            name="VIX", line=dict(color='#f6ad55', width=2),
            fill='tozeroy', fillcolor='rgba(246,173,85,0.08)',
        ))
        fig_vix.add_hline(y=30, line_dash="dash", line_color="#fc8181", annotation_text="Pánico (30)")
        fig_vix.add_hline(y=20, line_dash="dash", line_color="#f6ad55", annotation_text="Alerta (20)")
        fig_vix.add_hline(y=15, line_dash="dot",  line_color="#48bb78", annotation_text="Calma (15)")
        fig_vix.update_layout(**PLOTLY_DARK, height=250)
        st.plotly_chart(fig_vix, use_container_width=True)

        vix_val = df_ind['vix_level'].iloc[-1]
        if vix_val > 30:
            st.error(f"VIX en {vix_val:.1f} — pánico extremo.")
        elif vix_val > 20:
            st.warning(f"VIX en {vix_val:.1f} — volatilidad elevada.")
        else:
            st.success(f"VIX en {vix_val:.1f} — mercado en calma.")

    # ── 4. FEATURE IMPORTANCES ───────────────────────────────
    section("FEATURES · IMPORTANCIA DEL MODELO")

    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)

    col_fi1, col_fi2 = st.columns([2, 1])

    with col_fi1:
        top15  = imp.head(15).sort_values()
        fig_fi = go.Figure(go.Bar(
            x=top15.values, y=top15.index, orientation='h',
            marker=dict(color=top15.values, colorscale=[[0,'#2d3a8c'],[1,'#48bb78']]),
        ))
        fig_fi.update_layout(**PLOTLY_DARK, height=380)
        fig_fi.update_xaxes(gridcolor='#1e2535', title="Importancia")
        fig_fi.update_yaxes(gridcolor='#1e2535')
        st.plotly_chart(fig_fi, use_container_width=True)

    with col_fi2:
        st.markdown("**Por grupo**")
        group_imp = {g: imp[feats].sum() for g, feats in FEATURE_GROUPS.items()}
        group_s   = pd.Series(group_imp).sort_values(ascending=False)
        fig_pie   = go.Figure(go.Pie(
            labels=group_s.index, values=group_s.values, hole=0.4,
            marker=dict(colors=px.colors.sequential.Viridis),
            textfont=dict(size=10, family='IBM Plex Mono'),
        ))
        fig_pie.update_layout(
            paper_bgcolor='#0e1117', font=dict(color='#e2e8f0'),
            margin=dict(l=0, r=0, t=10, b=0), height=380,
            legend=dict(
                bgcolor='#1a1f2e',
                bordercolor='#2d3748',
                font=dict(size=11, color='#e2e8f0'),
            ),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── 5. SEÑALES ÚLTIMOS 30 DÍAS ───────────────────────────
    section("SEÑALES · ÚLTIMOS 30 DÍAS")

    last30 = bt.tail(30)[['Close','prob','pred','target']].copy()
    last30.columns = ['Close','P(Sube)','Predicción','Real']
    last30['Correcta']   = (last30['Predicción'] == last30['Real']).map({True:'✓', False:'✗'})
    last30['Predicción'] = last30['Predicción'].map({1:'↑ Sube', 0:'↓ Baja'})
    last30['Real']       = last30['Real'].map({1:'↑ Sube', 0:'↓ Baja'})
    last30['Close']      = last30['Close'].round(2)
    last30['P(Sube)']    = last30['P(Sube)'].round(3)
    st.dataframe(last30.iloc[::-1], use_container_width=True, height=300)

    # ── 6. TABLA ÚLTIMOS 20 DÍAS ─────────────────────────────
    section("MERCADO · ÚLTIMOS 20 DÍAS")

    df_recent = df.tail(21).copy()
    df_recent['Fecha']   = df_recent.index.strftime("%d/%m/%Y")
    df_recent['Cierre']  = df_recent['Close'].round(2)
    df_recent['Retorno'] = df_recent['Close'].pct_change().mul(100).round(2)
    df_recent['Subió']   = df_recent['target'].shift(1).map({1.0:'✅ Sí', 0.0:'❌ No'})

    tabla = df_recent[['Fecha','Cierre','Retorno','Subió']].tail(20)
    tabla = tabla.rename(columns={'Retorno':'Retorno (%)','Subió':'¿Subió al día siguiente?'})
    st.dataframe(tabla, use_container_width=True, hide_index=True)

    # ── FOOTER ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; color:#4a5568; font-size:12px; font-family:IBM Plex Mono,monospace;'>
      ⚠️ Solo fines educativos · No constituye asesoramiento financiero
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style='text-align:center; padding:80px 40px; color:#4a5568;'>
      <div style='font-size:48px; margin-bottom:16px;'>📊</div>
      <div style='font-family:IBM Plex Mono,monospace; font-size:14px; letter-spacing:0.1em;'>
        Configurá los parámetros en el panel izquierdo<br>
        y presioná <strong style='color:#718096;'>▶ Entrenar y evaluar</strong>
      </div>
    </div>
    """, unsafe_allow_html=True)