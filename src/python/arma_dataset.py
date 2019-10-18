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

#Aginación de estacion del año a registros
dataset = funciones.estacionalidad_y_eventos_extremos(dataset)

#Creacion de dataset para entrenar
x_data = []
y_data = []
nombres = []
x_data_columnas = ['caudal','precip_promedio','temp_max_promedio','Dia']
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

#Formateo de datos
ventana_de_tiempo = 24
espacio_de_tiempo = 24

x_train, y_train = funciones.sampling_temporal(x_train, y_train, ventana_de_tiempo, espacio_de_tiempo, flatten_data=False)
x_valid, y_valid = funciones.sampling_temporal(x_valid, y_valid, ventana_de_tiempo, espacio_de_tiempo, flatten_data=False)    
x_test, y_test = funciones.sampling_temporal(x_test, y_test, ventana_de_tiempo, espacio_de_tiempo,  flatten_data=False)    

x_train = np.concatenate(x_train)
x_valid = np.concatenate(x_valid)
x_test = np.concatenate(x_test)

y_train = np.concatenate(y_train).flatten()
y_valid = np.concatenate(y_valid).flatten()
y_test = np.concatenate(y_test).flatten()


#Escalado de datos
sc_x =  MinMaxScaler(feature_range=(0,1))
for i in range(x_train.shape[-1]):
    sc_x.fit_transform(x_train[:,:,i])
    x_train[:,:,i]= sc_x.transform(x_train[:,:,i])
    x_valid[:,:,i] = sc_x.transform(x_valid[:,:,i])
    x_test[:,:,i] = sc_x.transform(x_test[:,:,i])


#Guardado
np.savez(os.path.join(datadir, 'caudal_extra.npz'),
         x_train = x_train,
         x_valid = x_valid,
         x_test = x_test,
         y_train = y_train,
         y_valid = y_valid,
         y_test = y_test)