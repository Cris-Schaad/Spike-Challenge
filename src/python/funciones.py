import numpy as np
import pandas as pd
import itertools
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

    datos_estacion = pd.read_csv('caudal_extra.csv', header=0)
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

    datos_estacion = pd.read_csv('caudal_extra.csv', header=0)
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


def plot_eventos_extremos(nombre, variable, fechas, datos, labels):
 
    """
    Grafica registro historicos de variable distinguiendo eventos extremos 
    
    Parametros
    ----------
    nombre: str nombre estacion de medicion
    variable: str nombre de variable
    fechas: array 1-d con fechas de datos (datetime64[D])
    datos: array 1-d con datos
    labels: array 1-d con labels de datos (labels binarias: normal (0), evento extremo(1))

    -------
    """   
    
    plt.figure(figsize=(12,4))
    plt.scatter(fechas[np.where(~labels)], datos[np.where(~labels)], c='green', label='Normal', s=3)
    plt.scatter(fechas[np.where(labels)], datos[np.where(labels)], c='red', label='Evento extremo', s=3)
    plt.legend()
    plt.title('Registro histórico de '+variable+' para '+nombre)
    plt.xlabel('Fecha')
    plt.ylabel(variable)
    
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