# Regresión lineal y árboles de decisión para tareas de regresión

Laboratorio académico de la asignatura **Técnicas de Aprendizaje Automático**, correspondiente al **Máster Universitario en Inteligencia Artificial**. El proyecto aplica técnicas de aprendizaje automático supervisado para resolver una tarea de regresión usando un conjunto de datos de calidad del aire.

El análisis se desarrolla en un notebook de Python/Google Colab e incluye limpieza de datos, análisis exploratorio, entrenamiento de modelos, evaluación comparativa e investigación académica sobre aplicaciones reales de regresión.

## Objetivo del laboratorio

Aplicar los conocimientos de **regresión lineal** y **árboles de decisión para regresión** sobre un conjunto de datos real de calidad del aire, comprendiendo el tratamiento de datos necesario, la construcción de modelos predictivos y la interpretación de métricas de evaluación.

Los objetivos específicos son:

1. Comprender el tratamiento de datos previo al entrenamiento de modelos de regresión.
2. Aplicar regresión lineal simple y múltiple a un problema real.
3. Aplicar árboles de decisión para tareas de regresión.
4. Evaluar y comparar los resultados de los modelos.
5. Relacionar las técnicas estudiadas con casos reales documentados en literatura científica.

## Dataset utilizado

Se utilizó el archivo **`AirQualityUCI.csv`**, correspondiente al conjunto de datos de calidad del aire del repositorio UCI Machine Learning Repository.

El dataset contiene mediciones horarias de contaminantes atmosféricos, respuestas de sensores y variables meteorológicas. Entre sus variables se encuentran:

- `CO(GT)`: concentración de monóxido de carbono.
- `C6H6(GT)`: concentración de benceno.
- `NOx(GT)`: concentración de óxidos de nitrógeno.
- `NO2(GT)`: concentración de dióxido de nitrógeno.
- `PT08.S1(CO)`, `PT08.S2(NMHC)`, `PT08.S3(NOx)`, `PT08.S4(NO2)`, `PT08.S5(O3)`: respuestas de sensores químicos.
- `T`, `RH`, `AH`: temperatura, humedad relativa y humedad absoluta.
- `Date` y `Time`: variables temporales originales.

Después de la limpieza inicial se trabajó con **9.357 instancias útiles**.

## Variable objetivo

La variable respuesta seleccionada fue:

```text
C6H6(GT)
```

Esta variable representa la concentración horaria de benceno y es de tipo **numérico continuo**, por lo que corresponde a una tarea de **regresión**.

Se eligió porque permite estimar una concentración contaminante a partir de otras mediciones ambientales y de sensores, lo cual se relaciona directamente con problemas reales de monitoreo de calidad del aire.

## Tratamiento de datos

El notebook realiza las siguientes tareas de preparación:

1. Carga del archivo CSV con separador `;` y formato decimal europeo.
2. Eliminación de columnas vacías generadas por separadores finales.
3. Eliminación de filas totalmente vacías al final del archivo.
4. Reemplazo del valor `-200` por `NaN`, debido a que representa datos faltantes.
5. Conversión de `Date` y `Time` en una variable temporal `Timestamp`.
6. Creación de variables temporales derivadas:
   - `hour`
   - `dayofweek`
   - `month`
7. Selección de variables predictoras numéricas.
8. Imputación de valores faltantes mediante mediana.
9. Escalado de variables para los modelos lineales.
10. División de datos en conjuntos de entrenamiento y prueba.

## Análisis exploratorio de datos

Durante el análisis descriptivo se identificó que:

- El dataset contiene variables temporales, variables de concentración de gases, respuestas de sensores y variables meteorológicas.
- Existen valores faltantes codificados como `-200`.
- Algunas variables, como `NMHC(GT)`, presentan una proporción muy alta de datos faltantes.
- Las variables `CO(GT)`, `NOx(GT)` y `NO2(GT)` muestran asimetría positiva y posibles valores atípicos.
- Existen correlaciones altas entre ciertas respuestas de sensores y la variable objetivo `C6H6(GT)`.
- La variable más correlacionada con `C6H6(GT)` fue `PT08.S2(NMHC)`, por lo que se usó como predictor en la regresión lineal simple.

## Modelos implementados

### 1. Regresión lineal simple

Se entrenó un modelo de regresión lineal simple usando como predictor principal:

```text
PT08.S2(NMHC)
```

Este modelo permite analizar la relación directa entre una variable predictora altamente correlacionada y la concentración de benceno.

### 2. Regresión lineal múltiple

Se entrenó un modelo de regresión lineal múltiple usando varias variables predictoras numéricas, incluyendo sensores, contaminantes, variables meteorológicas y variables temporales derivadas.

El modelo se implementó mediante un pipeline con:

- `SimpleImputer(strategy='median')`
- `StandardScaler()`
- `LinearRegression()`

### 3. Árbol de decisión para regresión

Se entrenó un modelo `DecisionTreeRegressor` con los siguientes hiperparámetros:

```python
DecisionTreeRegressor(
    criterion='poisson',
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=2,
    max_features=None,
    random_state=42
)
```

Este modelo permite capturar relaciones no lineales entre las variables predictoras y la variable objetivo.

## Métricas de evaluación

Se utilizaron las siguientes métricas:

- **MAE**: error absoluto medio. Se interpreta en las mismas unidades de la variable objetivo.
- **RMSE**: raíz del error cuadrático medio. Penaliza más los errores grandes.
- **R²**: proporción de varianza explicada por el modelo.
- **RMSLE**: raíz del error logarítmico cuadrático medio. Útil para comparar errores relativos en variables no negativas.

La métrica principal considerada fue **MAE**, por ser interpretable y menos sensible a valores atípicos que RMSE.

## Resultados obtenidos

| Modelo | MAE | RMSE | R² | RMSLE |
|---|---:|---:|---:|---:|
| Regresión lineal simple | 1.0135 | 1.4148 | 0.9652 | 0.2382 |
| Regresión lineal múltiple | 0.8165 | 1.1312 | 0.9777 | 0.2078 |
| Árbol de decisión | 0.0171 | 0.1016 | 0.9998 | 0.0042 |

## Análisis de resultados

La regresión lineal simple obtuvo un buen desempeño usando únicamente `PT08.S2(NMHC)`, lo que confirma la fuerte relación entre esta variable y la concentración de benceno.

La regresión lineal múltiple mejoró los resultados de la regresión simple, ya que incorporó más información procedente de sensores, contaminantes, condiciones meteorológicas y variables temporales.

El árbol de decisión obtuvo el mejor desempeño global en la partición de prueba, con menor MAE, menor RMSE, menor RMSLE y mayor R². Sin embargo, este resultado debe interpretarse con cautela, ya que algunas variables predictoras están altamente correlacionadas con la variable objetivo. Por tanto, el modelo puede estar aprovechando relaciones muy directas entre sensores y concentración de benceno.

## Evaluación de sobreajuste

Para analizar el posible sobreajuste se compararon las métricas en entrenamiento y prueba.

| Modelo | Conjunto | MAE | RMSE | R² | RMSLE |
|---|---|---:|---:|---:|---:|
| Regresión lineal simple | Train | 1.0168 | 1.4077 | 0.9639 | 0.2552 |
| Regresión lineal simple | Test | 1.0135 | 1.4148 | 0.9652 | 0.2382 |
| Regresión lineal múltiple | Train | 0.7937 | 1.1180 | 0.9773 | 0.2154 |
| Regresión lineal múltiple | Test | 0.8165 | 1.1312 | 0.9777 | 0.2078 |
| Árbol de decisión | Train | 0.0104 | 0.1069 | 0.9998 | 0.0029 |
| Árbol de decisión | Test | 0.0171 | 0.1016 | 0.9998 | 0.0042 |

No se observó una señal clara de sobreajuste, ya que las métricas de entrenamiento y prueba son similares. Aun así, en un escenario real sería recomendable validar el modelo con validación cruzada y con datos temporales externos.

## Investigación académica incluida

El notebook incluye el análisis de un artículo científico posterior a 2015 que aplica modelos de regresión a un problema real de contaminación ambiental:

**Referencia APA:**

Bhathal, B. S., Gupta, G., & Sidhu, B. K. (2025). *Prediction of ambient air pollution with regression approach using machine learning*. **African Journal of Science, Technology, Innovation and Development, 17**(7), 994–1007. https://doi.org/10.1080/20421338.2025.2577980

El artículo aplica modelos de regresión para predecir el **Air Quality Index (AQI)** usando contaminantes como SO₂, CO, O₃, NO₂, PM2.5 y PM10. Compara técnicas como Linear Regression, Decision Tree Regression y Random Forest Regression. La investigación concluye que los modelos basados en árboles presentan ventajas para capturar relaciones no lineales en datos ambientales.

## Estructura sugerida del repositorio

```text
.
├── README.md
├── Actividad_1_resuelta_regresion_airquality_ejecutada.ipynb
├── AirQualityUCI.csv
├── Varios archivos .pdf
└── Rúbrica_MIA_TAA_act1.xlsx
```

## Requisitos

Para ejecutar el notebook se recomienda usar **Google Colab** o un entorno local con Python 3.10 o superior.

Librerías principales:

```text
numpy
pandas
matplotlib
scikit-learn
openpyxl
```

## Ejecución en Google Colab

1. Abrir el archivo `Actividad_1_resuelta_regresion_airquality_ejecutada.ipynb` en Google Colab.
2. Subir el archivo `AirQualityUCI.csv` al entorno de ejecución.
3. Ejecutar las celdas en orden.
4. Verificar la limpieza de datos, gráficos, entrenamiento de modelos y resultados finales.

## Ejecución local

Instalar dependencias:

```bash
pip install numpy pandas matplotlib scikit-learn openpyxl
```

Ejecutar Jupyter Notebook:

```bash
jupyter notebook Actividad_1_resuelta_regresion_airquality_ejecutada.ipynb
```

## Conclusión general

El laboratorio demuestra que el tratamiento de datos es una etapa fundamental antes de entrenar modelos de regresión. La conversión de valores perdidos, la selección de variables, la imputación, el escalado y la evaluación con métricas adecuadas son pasos esenciales para obtener resultados confiables.

En este caso, el árbol de decisión obtuvo los mejores resultados predictivos, aunque la regresión lineal múltiple también mostró un desempeño sólido y una interpretación más sencilla. La comparación entre modelos permite comprender que no solo importa la precisión, sino también la interpretabilidad, la complejidad y la validez de las relaciones aprendidas.

## Uso ético de IA generativa

Se utilizó IA generativa como herramienta de apoyo para organizar el análisis, completar el código base, mejorar la redacción académica y verificar la coherencia de las respuestas. Las métricas y resultados reportados provienen de la ejecución reproducible del notebook sobre el archivo CSV adjunto. No se fabricaron resultados ni referencias.

## Autor

Gabriel Eduardo Morejón López

## Licencia

Este repositorio tiene fines académicos y educativos. El uso del dataset debe respetar las condiciones del repositorio original de UCI Machine Learning Repository.
