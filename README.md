# Actividad 2.1: Estandarización de Pipelines y Control de Calidad de Datos

**Autor:** Julian Felipe Peralta Becerra  
**Institución:** Universidad Santo Tomás – Seccional Tunja  
**Versión:** 1.0.0 · 25 de noviembre de 2026

## 1. Objetivo

Dar continuidad al trabajo de preparación de datos de la actividad 1.1 con el conjunto Loan Default. Se implementa un flujo automatizado que verifica la calidad de las entradas, prepara los datos, entrena una regresión logística y evalúa su desempeño sobre registros reservados para prueba. También se define un contrato de datos para futuras inferencias.

## 2. Estructura del proyecto

```text
MLOps-Actividad-2-1/
├── app.py                    # Carga, control de calidad, entrenamiento y evaluación
├── pipeline.py               # Orquestación del flujo completo
├── validacion.py             # Validación sin dependencias externas
├── contrato_datos.json       # Reglas y ejemplos de inferencia
├── inferencia.py             # Validación y predicción de una entrada nueva
├── Makefile                  # Alternativa de automatización
├── requirements.txt          # Versiones de dependencias
├── README.md
├── .gitignore
├── data/
│   └── README.md             # Instrucciones para obtener el CSV original
├── tests/
│   └── test_validacion.py
└── resultados/
    ├── calidad_datos.json
    └── metricas.json
```

El archivo `modelo.joblib` se genera localmente durante el entrenamiento. No se requiere publicar el modelo binario, los entornos virtuales ni el CSV original.

## 3. Reproducción desde cero

Se verificó el proyecto con Python 3.13.5. Se recomienda Python 3.13 y las versiones exactas de `requirements.txt`. Desde la página de GitHub, descargar el proyecto con **Code > Download ZIP**, extraerlo y abrir una terminal en su carpeta principal. También se puede clonar el repositorio mediante Git si se tiene instalado.

Descargar el dataset original [Loan Default Dataset](https://www.kaggle.com/datasets/yasserh/loan-default-dataset) y colocar `Loan_Default.csv` dentro de `data/`. La copia utilizada en la verificación tiene 148.670 registros y 34 columnas. El CSV no se redistribuye en el repositorio; se conserva el enlace a la fuente. La huella SHA-256 de la copia utilizada se registra en `resultados/calidad_datos.json` para permitir verificar su identidad.

Crear un entorno virtual e instalar las dependencias.

**Windows (PowerShell o CMD):**

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe pipeline.py
```

**Linux o macOS:**

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python pipeline.py
```

Si el entorno virtual ya está activado, el comando único para ejecutar las pruebas, la validación de datos, el entrenamiento y la evaluación es:

```bash
python pipeline.py
```

Alternativamente, con GNU Make instalado y el entorno activo, ejecutar `make`. El archivo de datos también puede ubicarse en otra ruta mediante `python pipeline.py --data ruta/Loan_Default.csv`. La opción `--output` cambia la carpeta de resultados.

## 4. Contrato y control de calidad

El contrato `contrato_datos.json` centraliza los tipos, rangos, categorías y los ejemplos de entrada válida e inválida. Se seleccionaron siete variables de entrada: monto del préstamo, ingreso, puntaje crediticio, plazo, tipo de préstamo, propósito y tipo de ocupación. Se excluyen los identificadores, las demás columnas y la etiqueta `Status` de la inferencia. La etiqueta de entrenamiento es binaria (0/1).

La validación de `validacion.py` utiliza únicamente la biblioteca estándar de Python. Rechaza campos ausentes, nulos, tipos incorrectos, booleanos en campos numéricos, valores fuera del rango y categorías no admitidas. En inferencia también rechaza campos adicionales. En el CSV, se seleccionan las columnas del contrato, se convierten los textos numéricos a sus tipos y se comprueba cada fila antes de entrenar. Un entero como `360.0` en el CSV se acepta como 360; en JSON se exige un entero para los campos de ese tipo.

Los rangos son reglas operativas del ejercicio, no criterios oficiales de elegibilidad crediticia. Se permiten ingresos de cero y puntajes entre 300 y 900 para contemplar los valores del conjunto. Las filas que incumplen las reglas se excluyen y se contabilizan; no se sustituyen los valores nulos. Esta decisión puede introducir sesgo de selección y es una diferencia respecto a la imputación de la actividad 1.1. Solo se controla la calidad de las variables seleccionadas y la etiqueta; las demás columnas del CSV no intervienen en el modelo.

Para visualizar ambos ejemplos y ejecutar exclusivamente las pruebas:

```bash
python validacion.py
python -m unittest discover -s tests -v
```

## 5. Entrenamiento y evaluación

1. Comprobar el contrato mediante las pruebas unitarias. Si una prueba falla, el pipeline se detiene.
2. Leer el CSV y validar las variables de entrada y la etiqueta.
3. Dividir los datos válidos en 80 % para entrenamiento y 20 % para prueba, de forma estratificada y con semilla 42.
4. Estandarizar las variables numéricas y codificar las categóricas con One-Hot Encoding usando `ColumnTransformer` dentro de `Pipeline`. El ajuste se realiza únicamente con el conjunto de entrenamiento.
5. Entrenar la regresión logística con `class_weight='balanced'` para considerar el desequilibrio entre las clases.
6. Evaluar el modelo sobre prueba con accuracy, balanced accuracy, precisión, recall, F1, ROC-AUC y matriz de confusión. Se conserva un umbral de 0,5 y una comparación con la clase mayoritaria.
7. Guardar el reporte de calidad, métricas y modelo entrenado.

### Resultados de la ejecución verificada

| Indicador | Resultado |
|---|---:|
| Registros originales | 148.670 |
| Registros válidos | 139.353 |
| Registros rechazados | 9.317 |
| Entrenamiento | 111.482 |
| Prueba | 27.871 |
| Accuracy | 0,6663 |
| Balanced accuracy | 0,5758 |
| Precisión | 0,3567 |
| Recall | 0,3920 |
| F1-score | 0,3735 |
| ROC-AUC | 0,6014 |
| Accuracy de clase mayoritaria | 0,7463 |

Las 11 pruebas unitarias finalizaron correctamente. El modelo presenta una capacidad de discriminación limitada; su accuracy es inferior a la referencia de clase mayoritaria. Los pesos balanceados no garantizan una mejora en todas las métricas. El objetivo principal de este ejercicio es comprobar el funcionamiento reproducible del pipeline y sus controles de calidad, no obtener un sistema apto para aprobar o negar créditos. Se requiere una evaluación adicional de sesgos, disponibilidad temporal de variables, calibración y validación externa antes de cualquier uso real.

## 6. Inferencia

Después del entrenamiento se puede ejecutar una predicción con el ejemplo válido incorporado en el contrato:

```bash
python inferencia.py
```

Para utilizar un archivo JSON propio, copiar el objeto `ejemplo_valido` del contrato en un archivo `entrada.json` (sin la clave envolvente) y ejecutar:

```bash
python inferencia.py --entrada entrada.json
```

Si la entrada es inválida, el programa imprime los errores y finaliza con código 1 sin realizar ninguna predicción. Solo debe cargarse un modelo joblib generado localmente o procedente de una fuente de confianza. La probabilidad del modelo no equivale a una probabilidad de incumplimiento calibrada.

## 7. Fuente y alcance

- Loan Default Dataset: https://www.kaggle.com/datasets/yasserh/loan-default-dataset
- Documentación de Python: https://docs.python.org/3/
- Documentación de scikit-learn: https://scikit-learn.org/stable/

Proyecto académico para la actividad de MLOps. Los resultados se pueden reproducir con el mismo dataset, las dependencias declaradas y la semilla establecida. No se incluyen credenciales ni rutas absolutas de usuarios.

## 8. Referencias y material de apoyo

- TechWorld with Nana. (2025, 13 de noviembre). MLOps explained - what it is, why you need it and how it works [Video]. YouTube. https://www.youtube.com/watch?v=biqYkVf-a7Y

- Camp BI. (2025, 19 de septiembre). La mejor forma de crear pipelines de datos en Python [Video]. YouTube. https://www.youtube.com/watch?v=zmSo35zs_6U

- Pineau, J., et al. (2021). Improving reproducibility in machine learning research (a report from the NeurIPS 2019 reproducibility program). Journal of Machine Learning Research, 22(164), 1-20. https://www.jmlr.org/papers/volume22/20-303/20-303.pdf
