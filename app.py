import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.preprocessing import load_data, clean_data, split_temporal
from src.features import add_features, add_target, get_X_y, FEATURES
from src.model import (
    train_random_forest,
    predict,
    evaluate,
    plot_feature_importance,
)

# -----------------------------
# CONFIG STREAMLIT
# -----------------------------
st.set_page_config(
    page_title="SP500 Predictor",
    layout="wide"
)

st.title("📈 Predicción del S&P500 con Machine Learning")

st.write("""
Esta aplicación:
- carga datos históricos del S&P500
- genera features técnicas
- entrena un modelo Random Forest
- realiza predicciones
""")

# -----------------------------
# CARGAR DATASET
# -----------------------------
DATA_PATH = "data/sp500.csv"

try:
    df = load_data(DATA_PATH)
    df = clean_data(df)

    st.success("✅ Dataset cargado correctamente")

except Exception as e:
    st.error(f"Error cargando dataset: {e}")
    st.stop()

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
df = add_features(df)
df = add_target(df)

X, y = get_X_y(df)

# -----------------------------
# SPLIT TEMPORAL
# -----------------------------
X_train, X_test, y_train, y_test = split_temporal(X, y)

# -----------------------------
# ENTRENAMIENTO
# -----------------------------
st.subheader("🤖 Entrenando modelo...")

modelo = train_random_forest(X_train, y_train)

st.success("Modelo entrenado correctamente")

# -----------------------------
# PREDICCIONES
# -----------------------------
y_pred, y_proba = predict(modelo, X_test)

# -----------------------------
# MÉTRICAS
# -----------------------------
st.subheader("📊 Métricas del modelo")

metrics = evaluate(y_test, y_pred)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
col2.metric("Precision", f"{metrics['precision']:.4f}")
col3.metric("Recall", f"{metrics['recall']:.4f}")
col4.metric("F1 Score", f"{metrics['f1']:.4f}")

# -----------------------------
# RESULTADOS
# -----------------------------
st.subheader("📋 Predicciones")

results = pd.DataFrame({
    "Real": y_test.values,
    "Predicción": y_pred,
    "Probabilidad Suba": y_proba
}, index=y_test.index)

st.dataframe(results.tail(20))

# -----------------------------
# IMPORTANCIA DE FEATURES
# -----------------------------
st.subheader("🔥 Importancia de Variables")

fig, ax = plt.subplots(figsize=(10, 6))

importancias = pd.DataFrame({
    "feature": FEATURES,
    "importance": modelo.feature_importances_,
}).sort_values(by="importance", ascending=False)

ax.barh(importancias["feature"], importancias["importance"])
ax.invert_yaxis()

st.pyplot(fig)

# -----------------------------
# ÚLTIMA PREDICCIÓN
# -----------------------------
st.subheader("🎯 Última Predicción")

ultima_pred = y_pred[-1]
ultima_proba = y_proba[-1]

if ultima_pred == 1:
    st.success(
        f"El modelo predice SUBA 📈 "
        f"(probabilidad: {ultima_proba:.2%})"
    )
else:
    st.error(
        f"El modelo predice BAJA 📉 "
        f"(probabilidad: {1 - ultima_proba:.2%})"
    )