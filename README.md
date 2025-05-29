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

## 📊 Análisis Exploratorio

El análisis fue desarrollado siguiendo las buenas prácticas de EDA e incluye:

### ✅ Carga y limpieza de datos
- Revisión de tipos de datos, valores nulos y estadísticas básicas.

### ✅ Análisis univariado
- Distribución de cada variable individual.
- Histograma de calidad del vino.

![Histograma de calidad del vino](images/quality_distribution.png)

### ✅ Análisis bivariado
- Relación entre variables y la calidad del vino.
- Heatmap de correlaciones.

![Heatmap de correlaciones](images/heatmap_correlations.png)

### ✅ Hallazgos importantes
- El alcohol tiene una correlación positiva con la calidad.
- La acidez volátil se asocia negativamente con vinos de mejor calidad.
- Algunas variables tienen muy poca relación con la calidad.

---

## 🧪 Modelado (Opcional)

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
