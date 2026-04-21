# 📊 SP500 ML Prediction

## Descripción
Proyecto de Machine Learning orientado a la predicción de la dirección del índice S&P 500 en el corto plazo, utilizando datos históricos del mercado financiero. El objetivo es asistir en la toma de decisiones en trading intradiario mediante modelos predictivos basados en datos.


## Objetivo
Desarrollar un modelo de clasificación binaria que permita anticipar si el precio del S&P 500 subirá o bajará en el siguiente período, a partir de información histórica del mercado.


## Dataset
Se utiliza un dataset de datos históricos del índice S&P 500 con frecuencia horaria (H1), compuesto por:

- Open (precio de apertura)  
- High (precio máximo)  
- Low (precio mínimo)  
- Close (precio de cierre)  
- Volume (volumen de operaciones)  
- Cantidad de registros: ~3500  


## Enfoque del modelo
El problema se aborda como una tarea de **clasificación supervisada**.

Modelos a utilizar:
- Random Forest  
- XGBoost  

Se seleccionan estos algoritmos por su capacidad de manejar datos tabulares, capturar relaciones no lineales y ser robustos frente al ruido típico de los mercados financieros.


## Arquitectura del sistema

Datos históricos (OHLCV)
→ Preprocesamiento de datos
→ Feature Engineering (RSI, EMA, retornos, volatilidad)
→ Modelo de Machine Learning
→ Predicción (sube / baja + probabilidad)
→ Visualización en aplicación web (Streamlit)


## Tecnologías
- Python  
- Pandas / NumPy  
- Scikit-learn / XGBoost  
- Matplotlib / Seaborn  
- Streamlit  

## Estructura del repositorio

El proyecto se organiza en una estructura modular que permite separar datos, análisis y lógica del modelo:
•	data/: contiene los datos del proyecto, separados en datos crudos (raw) y procesados (processed).
•	notebooks/: incluye notebooks de análisis exploratorio y pruebas iniciales.
•	src/: contiene el código fuente del sistema, incluyendo preprocesamiento, generación de features y modelos.
•	README.md: documentación general del proyecto.
•	requirements.txt: librerías necesarias para reproducir el entorno.

Esta estructura permite separar las distintas etapas del pipeline de datos, facilitando la organización, mantenimiento y reproducibilidad del proyecto.

👉 

## Gestión del proyecto

Se utiliza GitHub Projects para la organización de tareas y seguimiento del avance:

👉 https://github.com/users/jrubio1912-oss/projects/1/views/1


## Estado del proyecto

- [x] Definición del problema  
- [x] Dataset identificado  
- [x] Arquitectura del sistema  
- [ ] Feature engineering  
- [ ] Entrenamiento del modelo  
- [ ] Evaluación del modelo  
- [ ] Desarrollo de aplicación (Streamlit)  


## Integrantes
- Choque Diaz, Diego Angel  
- Delgado, Julia  
- Dip, Julio  
- Llave, Ubaldo  
- Rubio, José  



