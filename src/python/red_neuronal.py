import funciones, os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras import optimizers
from sklearn.metrics import confusion_matrix


#importación datset en formato npz
thisdir = os.getcwd()
thisdir = Path(thisdir).parent.parent

plt.close('all')

dataset = np.load(os.path.join(thisdir, 'data', 'caudal_extra.npz'))
x_train = dataset['x_train']
y_train = dataset['y_train']
x_valid = dataset['x_valid']
y_valid = dataset['y_valid']
x_test = dataset['x_test']
y_test = dataset['y_test']
    

#Proporción de clases
prop_evento_extremo = sum(y_train)/len(y_train)
prop_evento_normal = 1-prop_evento_extremo


#Modelo de red neuronal
model = Sequential()
model.add(LSTM(units = 32, return_sequences = False, input_shape = x_train.shape[1:]))
model.add(Dense(units = 64, activation = 'sigmoid'))
model.add(Dense(units = 1, activation = 'sigmoid'))
model.summary()

adam = optimizers.adam(lr=0.001)
model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics=['accuracy'])

model_history = model.fit(x_train, y_train, batch_size = 256, epochs = 50, validation_data = (x_valid, y_valid), 
                          class_weight ={0:1/prop_evento_normal, 1:1/prop_evento_extremo}) #, verbose = 1)

#Gr'afico loss entrenamiento
plt.figure()
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Entrenamiento', 'Validación'], loc='upper right')


#Resultados
y_pred = model.predict(x_test)
cnf_matrix = confusion_matrix(y_test, y_pred.round())
funciones.plot_confusion_matrix(cnf_matrix, classes=['Normal','Evento Extremo'], normalize=True ,title='Caudales')
