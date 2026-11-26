import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, balanced_accuracy_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from validacion import CAMPOS, OBJETIVO, convertir_fila, validar_entrada

SEMILLA = 42
NUMERICAS = [c for c, r in CAMPOS.items() if r['tipo'] != 'string']
CATEGORICAS = [c for c, r in CAMPOS.items() if r['tipo'] == 'string']


def cargar_datos(ruta):
    registros = []
    errores_por_campo = Counter()
    clases_originales = Counter()
    descartados = 0
    total = 0
    with ruta.open(encoding='utf-8-sig', newline='') as archivo:
        lector = csv.DictReader(archivo)
        faltantes = (set(CAMPOS) | {OBJETIVO}) - set(lector.fieldnames or [])
        if faltantes:
            raise ValueError(f'Columnas faltantes: {sorted(faltantes)}')
        for fila in lector:
            total += 1
            datos = convertir_fila(fila)
            errores = validar_entrada(datos)
            estado = (fila.get(OBJETIVO) or '').strip()
            clases_originales[estado] += 1
            if estado not in {'0', '1'}:
                errores.append('Status: etiqueta no válida')
            if errores:
                descartados += 1
                errores_por_campo.update(errores)
                continue
            datos[OBJETIVO] = int(estado)
            registros.append(datos)
    df = pd.DataFrame(registros, columns=[*CAMPOS, OBJETIVO])
    reporte = {'filas_originales': total, 'filas_validas': len(df),
               'filas_rechazadas': descartados,
               'clases_originales': dict(clases_originales),
               'clases_validas': {str(k): int(v) for k, v in df[OBJETIVO].value_counts().items()},
               'incidencias': dict(errores_por_campo),
               'sha256_dataset': hashlib.sha256(ruta.read_bytes()).hexdigest()}
    if df.empty or df[OBJETIVO].value_counts().size != 2 or df[OBJETIVO].value_counts().min() < 5:
        raise ValueError('No hay suficientes registros válidos de ambas clases.')
    return df, reporte


def guardar_json(ruta, contenido):
    ruta.write_text(json.dumps(contenido, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def entrenar(ruta, salida):
    df, calidad = cargar_datos(ruta)
    X = df[list(CAMPOS)]
    y = df[OBJETIVO]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEMILLA, stratify=y)
    preparacion = ColumnTransformer([
        ('numericas', StandardScaler(), NUMERICAS),
        ('categoricas', OneHotEncoder(handle_unknown='ignore'), CATEGORICAS)
    ])
    modelo = Pipeline([
        ('preparacion', preparacion),
        ('clasificador', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEMILLA))
    ])
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    prob = modelo.predict_proba(X_test)[:, 1]
    metricas = {
        'modelo': 'Regresión logística con pesos balanceados',
        'semilla': SEMILLA, 'umbral': 0.5,
        'filas_entrenamiento': len(X_train), 'filas_prueba': len(X_test),
        'accuracy': accuracy_score(y_test, pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, pred),
        'precision': precision_score(y_test, pred, zero_division=0),
        'recall': recall_score(y_test, pred, zero_division=0),
        'f1': f1_score(y_test, pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, prob),
        'matriz_confusion': confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
        'accuracy_base_clase_mayoritaria': float((y_test == y_train.mode().iloc[0]).mean())
    }
    salida.mkdir(parents=True, exist_ok=True)
    guardar_json(salida / 'calidad_datos.json', calidad)
    guardar_json(salida / 'metricas.json', metricas)
    joblib.dump(modelo, salida / 'modelo.joblib')
    print(json.dumps(metricas, indent=2, ensure_ascii=False))
    return modelo
