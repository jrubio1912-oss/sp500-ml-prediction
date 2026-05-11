"""
model.py
Entrenamiento, predicción y evaluación del modelo Random Forest.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def train_random_forest(X_train, y_train, n_estimators: int = 100,
                        max_depth: int = 5, random_state: int = 42):
    """
    Entrena un RandomForestClassifier con class_weight='balanced'
    para compensar el desbalance entre clases típico en datos financieros.
    Retorna el modelo entrenado.
    """
    modelo = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight="balanced",
    )
    modelo.fit(X_train, y_train)
    return modelo


def predict(modelo, X_test):
    """
    Retorna las clases predichas y las probabilidades de la clase positiva.
    """
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]
    return y_pred, y_proba


def evaluate(y_test, y_pred) -> dict:
    """
    Calcula y muestra accuracy, precision, recall, F1-score
    y el classification report completo.
    Retorna un diccionario con las métricas principales.
    """
    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
    }

    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print()
    print(classification_report(y_test, y_pred))

    return metrics


def plot_confusion_matrix(y_test, y_pred) -> None:
    """Grafica la matriz de confusión."""
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicción")
    plt.ylabel("Valor real")
    plt.title("Matriz de confusión")
    plt.tight_layout()
    plt.show()


def plot_feature_importance(modelo, features: list) -> None:
    """Grafica la importancia de cada feature según el modelo."""
    importancias = pd.DataFrame({
        "feature":    features,
        "importance": modelo.feature_importances_,
    }).sort_values(by="importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importancias, x="importance", y="feature")
    plt.title("Importancia de variables en el modelo")
    plt.tight_layout()
    plt.show()

    return importancias

