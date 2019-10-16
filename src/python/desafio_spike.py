import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import funciones
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


plt.close('all')

##Importación de dataset
#dataset = pd.read_csv('caudal_extra.csv', usecols=[4,11,12,15,16], header=0)
#dataset = dataset.sort_values(by=['nombre', 'fecha'])
#
#print('Cantidad total de mediciones: ', dataset.shape[0])
#print('Cantidad de estaciones de medicion distintas: ', len(dataset.nombre.unique()))



#Analisis de mediciones por estación
#cantiadad_mediciones = []
#for i in dataset.nombre.unique():
#        cantiadad_mediciones.append(dataset[dataset['nombre'] == i].shape[0])
#        
#print('\nMinimma cantidad de datos registrados en una estación: ', np.min(cantiadad_mediciones))
#print('Máxima cantidad de datos registrados en una estación: ', np.max(cantiadad_mediciones))

##Analisis de datos faltantes
#print('\nDatos faltantes de caudal: ', dataset.caudal.isna().sum())
#print('Datos faltantes de precipitaciones: ', dataset.precip_promedio.isna().sum())
#print('Datos faltantes de temperatura máxima: ', dataset.temp_max_promedio.isna().sum())
#
#
#
##Analisis de fechas
#fechas_year = dataset.fecha.values.astype('datetime64[Y]')
#
#plt.figure()
#plt.hist(fechas_year, bins=59)


#dias_entre_mediciones = []
#for i in dataset.nombre.unique():
#    fechas_estacion = dataset[dataset['nombre']==i].fecha.values.astype('datetime64[D]')
#    dias_entre_mediciones = dias_entre_mediciones + list(np.diff(fechas_estacion))
#dias_entre_mediciones = np.asarray(dias_entre_mediciones).astype(int)
#
#plt.figure()
#plt.hist(dias_entre_mediciones, bins=[1,2,3,4,5])

##Analisis fechas de datos faltantes
#fechas_preci_faltantes = dataset[dataset['precip_promedio'].isna()].fecha.values.astype('datetime64[Y]')
#fechas_temps_faltantes = dataset[dataset['temp_max_promedio'].isna()].fecha.values.astype('datetime64[Y]')
#
#plt.figure()
#plt.hist(fechas_preci_faltantes, 50, alpha=0.5, facecolor='b')
#plt.hist(fechas_temps_faltantes, 50, alpha=0.5)





