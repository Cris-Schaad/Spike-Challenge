# Spike-Challenge

## Análsis de datos de caudales de rios en Chile: predicción de eventos extremos (Spike Labs)

Se requiere que el dataset "caudales.csv" esté en carpeta "data". En "src" están los códigos de ánalisis: se escribieron en python 3 y se realizaron copias para jupyter notebook.

Para el modelo de predicción de eventos extremos se utilizó una red LSTM. Esta toma las últimas 24 horas de medición para predecir lo que sucederá en las prózimas 24 horas. Se logra un 79% de precisión para los eventos normales y un 54% para los eventos extremos.

Para correr el modelo se necesitan las siguientes librerías:

- pandas
- numpy
- keras
- matplotlib
- scipy
- import_ipynb
