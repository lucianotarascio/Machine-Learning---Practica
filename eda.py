"""
21-09-2026
Análisis de datos exploratorios (EDA) utilizando el dataset iris.csv (medidas de sépalos y pétalos de flores de iris) 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar el dataset Iris desde un enlace público y confiable
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

# Inyectamos a propósito algunos valores nulos y anomalías para practicar limpieza
np.random.seed(42)
df.loc[10, 'sepal_length'] = np.nan       # Valor nulo
df.loc[25, 'petal_length'] = 15.5        # Valor atípico (outlier extremo)
df.loc[50, 'sepal_width'] = np.nan       # Valor nulo

print("--- DATOS ORIGINALES (Con nulos y anomalías inyectadas) ---")
display(df.head(15))

# 2. Limpieza de datos
# Rellenar valores nulos usando la mediana de cada especie o interpolación simple
df['sepal_length'] = df['sepal_length'].fillna(df['sepal_length'].median())
df['sepal_width'] = df['sepal_width'].fillna(df['sepal_width'].median())

# Filtrar o corregir el outlier de petal_length (por ejemplo, reemplazar valores imposibles > 10 con la media)
media_petal = df['petal_length'][df['petal_length'] < 10].mean()
df.loc[df['petal_length'] > 10, 'petal_length'] = media_petal

# 3. Detección simple de alertas basada en umbrales estadísticos o físicos
# Por ejemplo, marcar sépalos inusualmente largos (> 7.5 cm)
umbral_sepal = 7.5
df['alerta_sepal_largo'] = df['sepal_length'] > umbral_sepal

print("\n--- DATOS LIMPIOS Y REGISTROS CON ALERTA ---")
display(df[df['alerta_sepal_largo'] == True])

# Configuración visual
sns.set_theme(style="whitegrid")

# Cargar dataset Iris
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

print("--- 1. Información General del Dataset ---")
df.info()

print("\n--- 2. Métricas Estadísticas Descriptivas (Globales) ---")
display(df.describe())

print("\n--- 3. Métricas Agrupadas por Especie (Media y Desviación Estándar) ---")
metricas_especie = df.groupby('species').agg(
    promedio_sepal_largo=('sepal_length', 'mean'),
    desv_sepal_largo=('sepal_length', 'std'),
    promedio_petal_largo=('petal_length', 'mean'),
    desv_petal_largo=('petal_length', 'std'),
    total_muestras=('species', 'count')
).reset_index()

display(metricas_especie)

# 4. Visualización para el reporte y portfolio
plt.figure(figsize=(12, 5))

# Gráfico 1: Relación entre longitud de sépalo y pétalo por especie
plt.subplot(1, 2, 1)
sns.scatterplot(data=df, x='sepal_length', y='petal_length', hue='species', palette='Set2')
plt.title('Sépalo vs Pétalo por Especie')
plt.xlabel('Longitud de Sépalo')
plt.ylabel('Longitud de Pétalo')

# Gráfico 2: Distribución de la longitud del pétalo (Histograma con KDE)
plt.subplot(1, 2, 2)
sns.histplot(data=df, x='petal_length', hue='species', kde=True, palette='Set2', bins=20)
plt.title('Distribución de Longitud de Pétalo')
plt.xlabel('Longitud de Pétalo')

plt.tight_layout()
plt.show()
