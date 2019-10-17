import numpy as np
import pandas as pd
import os, itertools
from pathlib import Path
from scipy.stats import norm
import matplotlib.pyplot as plt


def time_plot_una_estacion(codigo_estacion, columna, fecha_min, fecha_max):

    """
    Grafica datos de 1 columna para 1 estación en intervalo de fechas definidas 
    por fecha_min y fecha_max

    Parametros
    ----------
    codigo_estacion : int
    columna: str, 'caudal', 'precip_promedio' o 'temp_max_promedio'
    fecha_min, fecha_max: str de fechas en formato 'YYYY-MM-dd'

    -------
    """    
    
    fecha_min = np.datetime64(str(fecha_min), 'h')
    fecha_max = np.datetime64(str(fecha_max), 'h')

    #Importación de dataset
    datadir = os.path.join(Path(os.getcwd()).parent.parent, 'data')
    datos_estacion = pd.read_csv(os.path.join(datadir, 'caudal_extra.csv'), header=0)
    datos_estacion = datos_estacion.sort_values(by=['nombre', 'fecha'])
    datos_estacion = datos_estacion[datos_estacion['codigo_estacion']==int(codigo_estacion)]
    
    #Seleccion de fecha
    datos_estacion = datos_estacion[datos_estacion['fecha'].astype('datetime64[h]')>=fecha_min]
    datos_estacion = datos_estacion[datos_estacion['fecha'].astype('datetime64[h]')<=fecha_max]
    
    x = datos_estacion.fecha.values.astype('datetime64[D]')
    y = datos_estacion[str(columna)]
        
    plt.figure()
    plt.plot(x,y)
    plt.title('Registros estación '+str(codigo_estacion))
    plt.ylabel(str(columna))
    plt.xlabel('Fecha')
    
    return None


def time_plot_estaciones_varias_columnas(codigo_estacion, columnas, fecha_min, fecha_max):
    
    """
    Grafica datos de columnas para 1 estación en intervalo de fechas definidas 
    por fecha_min y fecha_max, normalizados por el valor máximo de la serie

    Parametros
    ----------
    codigo_estacion : int
    columnas: lista de str, puede tener: 'caudal', 'precip_promedio' o 'temp_max_promedio'
    fecha_min, fecha_max: str de fechas en formato 'YYYY-MM-dd'

    -------
    """    
    
    fecha_min = np.datetime64(str(fecha_min), 'h')
    fecha_max = np.datetime64(str(fecha_max), 'h')

    #Importación de dataset
    datadir = os.path.join(Path(os.getcwd()).parent.parent, 'data')
    datos_estacion = pd.read_csv(os.path.join(datadir, 'caudal_extra.csv'), header=0)
    datos_estacion = datos_estacion.sort_values(by=['nombre', 'fecha'])
    datos_estacion = datos_estacion[datos_estacion['codigo_estacion']==int(codigo_estacion)]
    
    #Seleccion de fecha
    datos_estacion = datos_estacion[datos_estacion['fecha'].astype('datetime64[h]')>=fecha_min]
    datos_estacion = datos_estacion[datos_estacion['fecha'].astype('datetime64[h]')<=fecha_max]
    
    x = datos_estacion.fecha.values.astype('datetime64[D]')
        
    plt.figure()
    for i in columnas:
        y = datos_estacion[str(i)]/np.nanmax(datos_estacion[str(i)])
        plt.plot(x,y)
    plt.title('Registros estación '+str(codigo_estacion))
    plt.legend(columnas)
    plt.xlabel('Fecha')
    
    return None


def evento_extremo(x, interval_c = 0.95):
     
    """
    Determina valores en array x que estén fuera del intervalo de confianza (interval_c) 
    considerando distribución normal

    Asigna valor de 0 a elementos dentro del intervalo    
    Asigna valor de 1 a elementos fuera del intervalo

    Arguments
    ----------
    x : array 1-d
    interval_c: float entre 0 y 1

    -------
    Retorna
    
    array con labels para cada elemento del array x
    """    
    nan_index = np.isnan(x)
    x_sin_nan = x[~nan_index]    
    
    restos =  (1-interval_c)/2
    lim_menor = norm.ppf(restos)    
    lim_mayor = norm.ppf(interval_c+restos)
    x_norm = (x_sin_nan - np.mean(x_sin_nan))/np.std(x_sin_nan)
    
    labels = np.zeros(len(x))
    labels[~nan_index] = -1*(np.logical_and(x_norm >= lim_menor, x_norm <= lim_mayor).astype(int)-1)
    labels[nan_index] = np.nan
    
    return labels


def estacionalidad_y_eventos_extremos(dataset):

    """
    Determina la estacionalidad de los datos, y clasifica el tipo de dato (normal o extremo)

    Arguments
    ----------
    dataset: Pandas DataFrame

    -------
    Retorna
    
    dataset: Pandas DataFrame
    """    
    
    dataset['Verano'] = pd.Series(np.zeros(len(dataset)))
    dataset['Otoño'] = pd.Series(np.zeros(len(dataset)))
    dataset['Invierno'] = pd.Series(np.zeros(len(dataset)))
    dataset['Primavera'] = pd.Series(np.zeros(len(dataset)))
    dataset['Dia'] = pd.Series(np.zeros(len(dataset)))
    
    #Asignación de estacionalidad
    meses = pd.to_datetime(dataset['fecha']).dt.month.values
    dias = pd.to_datetime(dataset['fecha']).dt.dayofyear.values
    dataset.loc[dataset.index, 'Verano'] = np.logical_and(meses >= 1, meses <= 3).astype(int)
    dataset.loc[dataset.index, 'Otoño'] = np.logical_and(meses >= 4, meses <= 6).astype(int)
    dataset.loc[dataset.index, 'Invierno'] = np.logical_and(meses >= 7, meses <= 9).astype(int)
    dataset.loc[dataset.index, 'Primavera'] = np.logical_and(meses >= 10, meses <= 12).astype(int)
    dataset.loc[dataset.index, 'Dia'] = dias.astype(int)
    
    
    #Determinación de eventos extremos
    dataset['caudal_extremo'] = pd.Series(np.zeros(len(dataset)))
    dataset['precip_extremo'] = pd.Series(np.zeros(len(dataset)))
    dataset['temp_extremo'] = pd.Series(np.zeros(len(dataset)))
    
    
    for i in dataset.nombre.unique():
        datos_estacion = dataset[dataset['nombre'] == i]
        for est in ['Verano', 'Otoño', 'Invierno', 'Primavera']:
            datos = datos_estacion[datos_estacion[est]==1]
            dataset.loc[datos.index, 'caudal_extremo'] = evento_extremo(datos['caudal'].values)
            dataset.loc[datos.index, 'precip_extremo'] = evento_extremo(datos['precip_promedio'].values)
            dataset.loc[datos.index, 'temp_extremo'] = evento_extremo(datos['temp_max_promedio'].values)     
            
    return dataset


def plot_eventos_extremos_historicos(nombre_ejemplo, dataset, var):
 
    """
    Grafica registro historicos de variable distinguiendo eventos extremos 
    
    Parametros
    ----------
    nombre_ejemplo: str nombre estacion de medicion
    dataset: pandas DataFrame con categorización de eventos hecha
    var: str, puede ser 'Caudal', 'Precipitación promedio' o 'Temperatura máxima promedio'

    -------
    """   
    fechas = dataset[dataset['nombre']==nombre_ejemplo]['fecha'].values.astype('datetime64[D]')

    if var == 'Caudal':
        datos_ejemplo = dataset[dataset['nombre']==nombre_ejemplo]['caudal'].values
        labels_ejemplo = dataset[dataset['nombre']==nombre_ejemplo]['caudal_extremo'].values.astype(int)
    if var == 'Precipitación promedio':
        datos_ejemplo = dataset[dataset['nombre']==nombre_ejemplo]['precip_promedio'].values
        labels_ejemplo = dataset[dataset['nombre']==nombre_ejemplo]['precip_extremo'].values.astype(int)
    if var == 'Temperatura máxima promedio':
        datos_ejemplo = dataset[dataset['nombre']==nombre_ejemplo]['temp_max_promedio'].values
        labels_ejemplo = dataset[dataset['nombre']==nombre_ejemplo]['temp_extremo'].values.astype(int)

    plt.figure(figsize=(12,4))
    plt.scatter(fechas[np.where(~labels_ejemplo)], datos_ejemplo[np.where(~labels_ejemplo)], c='green', label='Normal', s=3)
    plt.scatter(fechas[np.where(labels_ejemplo)], datos_ejemplo[np.where(labels_ejemplo)], c='red', label='Evento extremo', s=3)
    plt.legend()
    plt.title('Registro histórico de '+var+' para '+nombre_ejemplo)
    plt.xlabel('Fecha')
    plt.ylabel(var)
    
    return None


def plot_eventos_acumulados(dataframe):
    
    """
    Grafica registro historicos de variable distinguiendo eventos extremos 
    
    Parametros
    ----------
    dataframe: dataframe de 2 columnas: la primera es de fechas, la segunda de 
    una de las variables de eventos extremos

    -------
    """      
    
    dataframe = dataframe.sort_values(by=['fecha'])    
    t = dataframe.iloc[:,0].values.astype('datetime64[D]')
    y = dataframe.iloc[:,1].values
    ylabel = dataframe.columns.values[-1]
    t = t[~np.isnan(y)]
    y = y[~np.isnan(y)].cumsum()
    y = y/np.linspace(0, len(y), len(y))
    
    plt.figure()
    plt.plot(t, y)
    plt.xlabel('Fecha')
    plt.ylabel(ylabel)
    plt.title('Proporción de eventos de '+str(ylabel))  
    
    return None


def sampling_temporal(x, y, ventana_de_tiempo, espacio_de_tiempo, time_step = 1, flatten_data=True):
 
    """
    Divide los datos (x) en segmentos de tamaño definido por ventana_de_tiempo, con un pasos definidos por time_step.
    Las labels correspondientes (y) a cada segmento 
    
    Parametros
    ----------
    x: array N-d de datos
    y: array N-d de labels
    ventana de tiempo: int, tamaño de segmentos
    espacio_de_tiempo: int, espacion de tiempo entre segmento y label
    time_step: int, salto entre segmentos
    flatten_data: boolean, flattening de segmentos
    
    Retorna
    -------
    x, y sampleados
    """       
    x_sampleado = []
    y_sampleado = []
    for i in range(x.shape[0]):
        x_sample = x[i]
        y_sample = y[i]

        indx = len(x_sample)-1
        indx_left = True
                
        sampled_x = []
        sampled_y = []
        while indx_left:
            x_ventana = x_sample[indx - espacio_de_tiempo - ventana_de_tiempo:indx - espacio_de_tiempo]
            if flatten_data:
                x_ventana = x_ventana.flatten()
            y_ventana = y_sample[indx]
            if np.sum(np.isnan(x_ventana)) == 0:
                sampled_x.append(x_ventana)
                sampled_y.append(y_ventana)
            indx = indx-time_step
            if indx - espacio_de_tiempo - ventana_de_tiempo < 0:
                indx_left=False
        
        if len(sampled_x) > 0:
            x_sampleado.append(np.asarray(sampled_x))
            y_sampleado.append(np.asarray(sampled_y))
          
    x_sampleado = np.asarray(x_sampleado)
    y_sampleado = np.asarray(y_sampleado)
    
    return x_sampleado, y_sampleado


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()