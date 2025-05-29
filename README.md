# 🍷 Wine Quality EDA - Análisis Exploratorio del Dataset de Calidad del Vino

Este repositorio contiene un análisis exploratorio de datos (EDA) realizado sobre el dataset de *Wine Quality (Red Wine)*, disponible en el [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality). Este proyecto fue desarrollado con el objetivo de demostrar habilidades en análisis de datos, visualización, y comunicación de hallazgos.

> También puedes modificar fácilmente el código para trabajar con el vino blanco o ambas muestras combinadas.

---

## 📁 Contenido

- `wine_data_analysis.ipynb`: Notebook de Google Colab con el análisis completo.
- `images/`: Carpeta con visualizaciones del análisis (gráficos generados desde el notebook).
- `data/`: Carpeta opcional para incluir el dataset si se desea.

---

## 🔍 Descripción del dataset

Los datasets contienen información sobre variantes tintas y blancas del vino portugués "Vinho Verde". Incluyen variables fisicoquímicas y una evaluación sensorial de calidad (escala de 0 a 10). Se pueden abordar como problemas de regresión o clasificación, aunque las clases no están balanceadas. No se dispone de datos comerciales o de procedencia. Dada la naturaleza del dataset, es útil aplicar detección de outliers y métodos de selección de características.

**Características**:
- 1599 muestras de vino tinto.
- 11 variables físico-químicas como acidez, azúcar residual, pH, alcohol, entre otros.
- Una variable objetivo: `quality`.

---
### 🍷 Flexibilidad en el análisis: vino tinto, blanco o ambos

Este proyecto realiza un análisis exploratorio sobre el dataset **Wine Quality** del repositorio UCI, enfocado por defecto en el **vino tinto** (*red wine*). No obstante, el código ha sido diseñado de forma flexible para que puedas cambiar fácilmente el tipo de vino a analizar.

En la sección `🚀 Cargar dataset` del notebook, puedes modificar el valor de la variable `variant` según el análisis deseado:

```python
# 🚀 Cargar dataset
wine_quality = load_wine_dataset()

# Selecciona:
# [1] para análisis de vino tinto
# [2] para análisis de vino blanco
# [3] para combinar ambos (tinto y blanco)
variant = 1  # Cambia este valor según tu interés
```
## 📊 Análisis Exploratorio

El análisis fue desarrollado siguiendo las buenas prácticas de EDA e incluye:

### ✅ Carga y limpieza de datos
- Revisión de tipos de datos, valores nulos y estadísticas básicas.

 ![estadísticas básicas](images/describe.png)
 ![datos nulos](images/info.png)

No hay datos faltantes y podemos decir que hay unos valores atípicos los cuales es bueno analizar.

### 🔎 Agrupación de la calidad

Para facilitar el modelado y el análisis de clases balanceadas, se agruparon los valores de la variable quality:

- 0 = Baja calidad (calificaciones de 3 a 4)
- 1 = Media calidad (calificaciones de 5 a 6)
- 2 = Alta calidad (calificaciones de 7 a 8)

Esta clasificación mejora la interpretación y balancea mejor las clases para modelos supervisados.
![Histograma de calidad del vino](images/quality_distribution.png)
![agrupados](images/quality2.png)


#### ✅ Tratamiento de outliers

Para identificar datos atípicos (outliers), se aplicó el método del Z-modificado (Zmod) utilizando la mediana y la desviación absoluta mediana (MAD). Este enfoque es más robusto frente a distribuciones sesgadas o con valores extremos en comparación con el Z-score clásico.

Se detectaron valores atípicos en varias variables numéricas, y estos fueron revisados para su posterior tratamiento (eliminación o análisis por separado).
```python
def detectar_atipicos_zmod(df, umbral=3.5):
    outliers = {}

    for col in df.select_dtypes(include='number'):
        x = df[col]
        mediana = x.median()
        mad = np.median(np.abs(x - mediana))

        if mad == 0:
            continue  # Evitar división por cero

        zmod = 0.6745 * (x - mediana) / mad
        mask_outliers = np.abs(zmod) > umbral
        outliers[col] = df[mask_outliers][col]

    return outliers
```
 ![valores atípicos](images/atipicos.png)
 


 
### ✅ Análisis univariado
Durante el análisis univariado se examinaron las distribuciones individuales de cada variable físico-química presente en el dataset. A partir de estas gráficas se observó que ciertas variables tienden a agruparse alrededor de valores específicos para cada rango de quality, lo cual sugiere que es posible establecer diferencias entre vinos de distinta calidad basándose únicamente en algunas de estas características.

Este comportamiento indica que variables como el alcohol, el pH, los sulfitos y la densidad podrían ser útiles para **diferenciar entre tipos de vino** y predecir su calidad de manera preliminar.

Entre las variables más relevantes que mostraron patrones claros en su distribución se encuentran:

-**Alcohol vs pH**
 ![grafica 1](images/alcoholVSph.png)

-**Alcohol vs Sulphates**
 ![grafica 2](images/sulphates.png)
 
-**Density vs Total Sulfur Dioxide**
 ![grafica 3](images/densityVStotalsulfurdioxide.png)

-**Alcohol vs Free Sulfur Dioxide**
 ![grafica 4](images/alcoholVSfreesulfurdioxide.png)

Estas gráficas permitieron identificar rangos y comportamientos típicos de vinos de mayor o menor calidad, sentando las bases para un análisis bivariado más profundo y una futura etapa de modelado predictivo.

### 🧪 Ingeniería de características
Se crearon nuevas variables derivadas a partir de las características originales, para explorar su impacto en la predicción de calidad del vino:

```python
# Nuevas características derivadas
df['alcohol_density'] = df['alcohol'] * df['density']
df['acidez_ratio'] = df['fixed_acidity'] / (df['volatile_acidity'] + 1e-5)
df['dilucion_efecto'] = df['residual_sugar'] / (df['density'] + 1e-5)
df['sulfur_ratio'] = df['total_sulfur_dioxide'] / (df['free_sulfur_dioxide'] + 1e-5)
df['score_equilibrio'] = (df['alcohol'] * df['sulphates']) / ((df['volatile_acidity'] + 1e-5) * (df['chlorides'] + 1e-5))
df['inv_density'] = 1 / (df['density'] + 1e-5)
```
Estas variables permiten capturar interacciones no lineales y relaciones más complejas entre los compuestos químicos y la calidad del vino.  


### ✅ Análisis bivariado

En esta etapa se exploraron las relaciones entre pares de variables para entender mejor cómo interactúan entre sí y cómo estas interacciones se relacionan con la variable objetivo `quality`.

A partir del análisis bivariado se detectaron correlaciones interesantes que confirman y complementan lo observado en el análisis univariado. Las visualizaciones empleadas, como diagramas de dispersión y mapas de calor, permitieron identificar tendencias y patrones relevantes.

Algunas de las relaciones más destacadas incluyen:

- **Alcohol vs Quality**: Existe una correlación positiva clara, donde los vinos con mayor contenido de alcohol tienden a tener una mejor calidad sensorial.
- **Acidez volátil vs Quality**: Se observa una relación negativa, indicando que niveles altos de acidez volátil están asociados a vinos de menor calidad.
- **Sulphates vs Quality**: Se evidenció una ligera correlación positiva, mostrando que los sulfitos pueden influir en la percepción del vino.
- **Density vs Alcohol**: Presentan una relación inversa, lo cual puede ser útil al momento de seleccionar características para modelado.
- **Free Sulfur Dioxide vs Total Sulfur Dioxide**: Estas dos variables están altamente correlacionadas, lo que puede llevar a considerar eliminar una de ellas en análisis posteriores para evitar redundancia.

Además, se generó un **heatmap de correlación** entre todas las variables numéricas, que sirvió como guía para detectar relaciones fuertes y posibles redundancias, así como para seleccionar variables clave que podrían tener un mayor peso en un modelo predictivo.

Estas relaciones serán fundamentales para los siguientes pasos del análisis y permitirán construir modelos más interpretables y precisos.


![Heatmap de correlaciones](images/heatmap_correlations.png)

## 🔍 Análisis de Correlación - Mapa de Calor

A continuación, se presenta el análisis de correlación utilizando un mapa de calor que permite visualizar la relación entre las diferentes variables del dataset:

### 📈 Conclusiones del Mapa de Correlación

#### ✅ Relaciones fuertes positivas

- **`fixed_acidity` y `citric_acid`** (**0.69**): Existe una fuerte relación positiva. A mayor acidez fija, mayor es también el contenido de ácido cítrico.
- **`residual_sugar` y `dilucion_efecto`** (**1.00**): Relación perfecta, probablemente una es derivada de la otra. Se recomienda eliminar una para evitar multicolinealidad.
- **`free_sulfur_dioxide` y `total_sulfur_dioxide`** (**0.64**): El dióxido de azufre total incluye el libre, lo que justifica esta relación.
- **`alcohol` y `inv_density`** (**0.55**) y relación negativa con `density` (**-0.55**): El contenido de alcohol reduce la densidad del vino, lo cual es coherente.

#### ❌ Relaciones fuertes negativas

- **`fixed_acidity` y `pH`** (**-0.71**): A mayor acidez, menor pH. Relación química esperada.
- **`acidez_ratio` y `volatile_acidity`** (**-0.81**): Cuando la acidez volátil aumenta, el ratio general disminuye, indicando vinos posiblemente de menor calidad.
- **`density` y `inv_density`** (**-1.00**): Relación perfecta inversa. Son recíprocas matemáticamente.

#### 🏅 Relación con la calidad (`quality`)

- **`alcohol`** (**0.44**): Es la variable más correlacionada positivamente con la calidad. Vinos con mayor contenido alcohólico tienden a tener mejor puntuación.
- **`volatile_acidity`** (**-0.39**): Alta acidez volátil tiende a reducir la calidad del vino.
- **`sulphates`** (**0.27**): Leve correlación positiva. Puede contribuir a una mejor percepción del vino.

#### 📌 Otras observaciones

- Las variables derivadas como `alcohol_density`, `score_equilibrio` y `acidez_ratio` muestran correlaciones relevantes con variables originales. Pueden ser útiles, pero se debe revisar la **multicolinealidad**.
- Algunas variables como `chlorides` y `residual_sugar` tienen correlaciones muy bajas con la calidad, lo que indica baja relevancia para modelos predictivos.

### 🧠 Recomendaciones

- **Eliminar variables duplicadas o altamente correlacionadas**, como `density`/`inv_density` o `residual_sugar`/`dilucion_efecto`.
- **Conservar variables clave** como `alcohol`, `volatile_acidity` y `sulphates`, dada su relevancia en la predicción de calidad.
- **Revisar la utilidad de variables derivadas**, evaluando si realmente aportan información adicional significativa.

---

Este análisis proporciona una base sólida para la selección de características en futuras etapas de modelado predictivo.


---

## 🧪 Modelado 

El proyecto también incluye una sección opcional donde se prepara el dataset para modelado (normalización, selección de características), permitiendo probar algoritmos de regresión o clasificación como regresión logística, árbol de decisión, etc.

---

## 🛠 Herramientas utilizadas

- Python 3.10+
- Google Colab
- Pandas
- Matplotlib / Seaborn
- Scikit-learn (para modelado opcional)

---

## 📁 Cómo usar este proyecto

Puedes abrir directamente el notebook en Google Colab:

📎 [Ver en Google Colab](https://colab.research.google.com/drive/1lhzlRoNkSfiBZdrXBTcqROA7cgX-qd-K)

O clonarlo localmente:

```bash
git clone https://github.com/tu_usuario/wine-quality-eda.git
