# 📊 SP500 ML Prediction

## Descripción
Proyecto de Machine Learning orientado a la predicción de la dirección del índice S&P 500 al día siguiente, utilizando datos históricos descargados en tiempo real. El objetivo es asistir en la toma de decisiones mediante modelos predictivos basados en datos técnicos y macroeconómicos.

## 🧠 ¿Qué hace este sistema? (resumen no técnico)

Este sistema analiza el comportamiento histórico del mercado de acciones de Estados Unidos y genera una predicción sobre si el S&P 500 **subirá o bajará al día siguiente**.

Para hacer esa predicción, el modelo analiza automáticamente 38 indicadores del mercado:
- Cómo se movió el precio en los últimos días, semanas y meses
- Qué tan nervioso está el mercado (índice VIX)
- Cómo están las tasas de interés y el dólar
- Señales técnicas como el RSI y el MACD que usan los analistas financieros

El resultado se presenta como una señal (**↑ Sube** o **↓ Baja**) junto con una probabilidad de ocurrencia.

### ¿Qué tan confiable es?

El modelo acierta aproximadamente el **51-53% de las veces**. Esto puede parecer poco, pero en mercados financieros tiene un significado importante:

- Predecir mercados es extremadamente difícil — los precios incorporan información de millones de operadores en tiempo real
- Una pequeña ventaja consistente sobre el azar (similar a un casino que gana el 51% de las apuestas) puede generar valor a largo plazo
- El modelo **no es un oráculo** — es una herramienta de apoyo, no un sistema automático de inversión

> ⚠️ Este proyecto tiene fines exclusivamente educativos. No constituye asesoramiento financiero ni garantiza rentabilidad.

---

## Objetivo
Desarrollar un modelo de clasificación binaria que permita anticipar si el precio del S&P 500 subirá o bajará al día siguiente, a partir de 38 features técnicos y macroeconómicos.

## Dataset
Datos históricos diarios descargados en tiempo real vía `yfinance`:

- **S&P 500** (^GSPC) — OHLCV diario
- **VIX** (^VIX) — índice de volatilidad
- **Tasas 10Y** (^TNX) — rendimiento del bono del Tesoro
- **DXY** (DX-Y.NYB) — índice del dólar
- **Oro** (GC=F) — precio del oro
- Historia: hasta 20 años de datos diarios

## Features del modelo (38 en total)

| Grupo | Features |
|---|---|
| Retornos | 1d, 3d, 5d, 10d, 21d |
| Volatilidad | 5d, 21d, 63d + ratio |
| Volumen | normalizado 5d y 21d |
| Tendencia | distancia a EMA 5/21/63/200 |
| Momentum | RSI 14, MACD, Bollinger Bands |
| Calendario | día de la semana, mes |
| VIX | nivel, retornos, ratio MA, flag pánico |
| Tasas | nivel, cambios, pendiente |
| DXY | retornos 1d y 5d |
| Oro | retornos 1d y 5d |
| Correlaciones | SP500 vs VIX, SP500 vs Oro (21d) |

## Modelo
**LightGBM** con optimización de hiperparámetros y validación temporal estricta (walk-forward). Se seleccionó por su velocidad, manejo eficiente de datos tabulares y rendimiento superior en series temporales financieras.

## Arquitectura del sistema

```
Datos en tiempo real (yfinance)
  → Preprocesamiento (src/preprocessing.py)
  → Feature Engineering (src/features.py)
  → Modelo LightGBM (src/model.py)
  → Predicción + Backtest + Visualización (app.py)
```

## Tecnologías

- Python 3.11
- LightGBM / Scikit-learn
- Pandas / NumPy
- Streamlit
- Plotly
- yfinance
- Docker

## Estructura del repositorio

```
├── src/
│   ├── preprocessing.py   # Descarga de datos vía yfinance
│   ├── features.py        # Feature engineering (38 features)
│   └── model.py           # Entrenamiento, backtest y predicción
├── notebooks/             # Análisis exploratorio y modelado inicial
├── data/                  # Datos crudos y procesados
├── app.py                 # App Streamlit
├── requirements.txt
├── .gitignore
├── Dockerfile
└── .streamlit/
    └── config.toml
```

---

## ▶ Cómo correr la app

### Opción 1 — Docker (recomendado, sin instalar nada)

Requiere tener [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado y corriendo.

```bash
docker run -p 8501:8501 jrubio1912/sp500-predictor
```

Luego abrí el navegador en `http://localhost:8501`

---

### Opción 2 — Python local

**1. Clonar el repositorio**
```bash
git clone https://github.com/jrubio1912-oss/sp500-ml-prediction.git
cd sp500-ml-prediction
```

**2. Crear entorno virtual**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Instalar dependencias**
```bash
pip install -r requirements.txt
```

**4. Correr la app**
```bash
streamlit run app.py
```

Luego abrí el navegador en `http://localhost:8501`

---

## Uso de la app

1. Configurá los parámetros en el panel izquierdo (años de entrenamiento, fecha de corte, hiperparámetros)
2. Presioná **▶ Entrenar y evaluar**
3. La app descarga datos frescos, entrena el modelo y muestra:
   - Señal del próximo día hábil (↑ Sube / ↓ Baja) con probabilidad
   - Métricas del modelo (Accuracy, AUC-ROC, Sharpe, Max Drawdown)
   - Gráfico de velas con indicadores seleccionables
   - Tabs de RSI, MACD y VIX
   - Feature importances
   - Backtest: estrategia vs buy & hold
   - Señales de los últimos 30 días

---

## Estado del proyecto

- [x] Definición del problema
- [x] Dataset identificado
- [x] Feature engineering (38 features)
- [x] Entrenamiento del modelo (LightGBM)
- [x] Evaluación y backtest
- [x] Aplicación Streamlit
- [x] Dockerización
- [ ] Deploy en la nube

---

## Gestión del proyecto

👉 [GitHub Projects](https://github.com/users/jrubio1912-oss/projects/1/views/1)

---

