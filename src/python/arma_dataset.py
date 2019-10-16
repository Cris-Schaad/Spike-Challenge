import numpy as np
import pandas as pd
import funciones, os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


#Importación de dataset
datadir = os.path.join(Path(os.getcwd()).parent.parent, 'data')
dataset = pd.read_csv(os.path.join(datadir, 'caudal_extra.csv'), usecols=[4,11,12,15,16], header=0)
dataset = dataset.sort_values(by=['nombre', 'fecha'])

#Aginación de estacionalidad del año a medidas
dataset['Verano'] = pd.Series(np.zeros(len(dataset)))
dataset['Otoño'] = pd.Series(np.zeros(len(dataset)))
dataset['Invierno'] = pd.Series(np.zeros(len(dataset)))
dataset['Primavera'] = pd.Series(np.zeros(len(dataset)))


meses = pd.to_datetime(dataset['fecha']).dt.month.values
dataset.loc[dataset.index, 'Verano'] = np.logical_and(meses >= 1, meses <= 3).astype(int)
dataset.loc[dataset.index, 'Otoño'] = np.logical_and(meses >= 4, meses <= 6).astype(int)
dataset.loc[dataset.index, 'Invierno'] = np.logical_and(meses >= 7, meses <= 9).astype(int)
dataset.loc[dataset.index, 'Primavera'] = np.logical_and(meses >= 10, meses <= 12).astype(int)


#Determinación de eventos extremos
dataset['caudal_extremo'] = pd.Series(np.zeros(len(dataset)))
dataset['precip_extremo'] = pd.Series(np.zeros(len(dataset)))
dataset['temp_extremo'] = pd.Series(np.zeros(len(dataset)))


for i in dataset.nombre.unique():
    datos_estacion = dataset[dataset['nombre'] == i]
    for est in ['Verano', 'Otoño', 'Invierno', 'Primavera']:
        datos = datos_estacion[datos_estacion[est]==1]
        dataset.loc[datos.index, 'caudal_extremo'] = funciones.evento_extremo(datos['caudal'].values)
        dataset.loc[datos.index, 'precip_extremo'] = funciones.evento_extremo(datos['precip_promedio'].values)
        dataset.loc[datos.index, 'temp_extremo'] = funciones.evento_extremo(datos['temp_max_promedio'].values)     


#Graficos eventos extremos
nombre_ejemplo = 'Rio Turbio En Varillar'
fechas = dataset[dataset['nombre']==nombre_ejemplo]['fecha'].values.astype('datetime64[D]')
caudal_ejemplo = dataset[dataset['nombre']==nombre_ejemplo]['caudal'].values
precip_ejemplo = dataset[dataset['nombre']==nombre_ejemplo]['precip_promedio'].values
temp_ejemplo = dataset[dataset['nombre']==nombre_ejemplo]['temp_max_promedio'].values

labs_c = dataset[dataset['nombre']==nombre_ejemplo]['caudal_extremo'].values.astype(int)
labs_p = dataset[dataset['nombre']==nombre_ejemplo]['precip_extremo'].values.astype(int)
labs_t = dataset[dataset['nombre']==nombre_ejemplo]['temp_extremo'].values.astype(int)

funciones.plot_eventos_extremos(nombre_ejemplo, 'caudal', fechas, caudal_ejemplo, labs_c)
funciones.plot_eventos_extremos(nombre_ejemplo, 'precipitación promedio', fechas, precip_ejemplo, labs_p)
funciones.plot_eventos_extremos(nombre_ejemplo, 'temperatura maxima promedio', fechas, temp_ejemplo, labs_t)


#Creacion de dataset para entrenar
x_data = []
y_data = []
nombres = []
x_data_columnas = ['caudal','precip_promedio','temp_max_promedio','Verano','Otoño','Invierno','Primavera']
y_data_columnas = ['caudal_extremo']


#Creación de dataset para red neuronal en formato .npz
for i in dataset.nombre.unique():
    datos_estacion = dataset[dataset['nombre'] == i]    
    nombres.append(i)
    x_data.append(datos_estacion[x_data_columnas].values)
    y_data.append(datos_estacion[y_data_columnas].values)    

x_data = np.asarray(x_data)   
y_data = np.asarray(y_data)   


#Division de dataset
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)


#Escalado de datos
sc_x =  MinMaxScaler(feature_range=(0,1))
sc_x.fit_transform(np.concatenate(x_train))

for sample in range(len(x_train)):
    x_train[sample] = sc_x.transform(x_train[sample])
for sample in range(len(x_valid)):
    x_valid[sample] = sc_x.transform(x_valid[sample])
for sample in range(len(x_test)):
    x_test[sample] = sc_x.transform(x_test[sample])


#Formateo de datos
ventana_de_tiempo = 7
espacio_de_tiempo = 1

x_train, y_train = funciones.sampling_temporal(x_train, y_train, ventana_de_tiempo, espacio_de_tiempo, flatten_data=False)
x_valid, y_valid = funciones.sampling_temporal(x_valid, y_valid, ventana_de_tiempo, espacio_de_tiempo, flatten_data=False)    
x_test, y_test = funciones.sampling_temporal(x_test, y_test, ventana_de_tiempo, espacio_de_tiempo,  flatten_data=False)    


x_train = np.concatenate(x_train)
x_valid = np.concatenate(x_valid)
x_test = np.concatenate(x_test)

y_train = np.concatenate(y_train).flatten()
y_valid = np.concatenate(y_valid).flatten()
y_test = np.concatenate(y_test).flatten()

#Guardado
np.savez(os.path.join(datadir, 'caudal_extra.npz'),
         x_train = x_train,
         x_valid = x_valid,
         x_test = x_test,
         y_train = y_train,
         y_valid = y_valid,
         y_test = y_test)