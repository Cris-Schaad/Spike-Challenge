import funciones
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn.metrics import confusion_matrix

plt.close('all')

dataset = np.load('caudal_extra.npz')
x_train = dataset['x_train']
y_train = dataset['y_train']
x_valid = dataset['x_valid']
y_valid = dataset['y_valid']
x_test = dataset['x_test']
y_test = dataset['y_test']
    

#Proporción de clases
prop_evento_extremo = sum(y_train)/len(y_train)
prop_evento_normal = 1-prop_evento_extremo


model = Sequential()
model.add(Dense(units = 256, activation = 'sigmoid', input_shape = x_train[0].shape))
model.add(Dense(units = 256, activation = 'sigmoid'))
model.add(Dense(units = 1, activation = 'sigmoid'))

adam = optimizers.adam(lr=0.01)
model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics=['accuracy'])

model_history = model.fit(x_train, y_train, batch_size = 256, epochs = 20, validation_data = (x_valid, y_valid), 
                          class_weight ={0:1/prop_evento_normal, 1:1/prop_evento_extremo}) #, verbose = 1)

plt.figure()
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Entrenamiento', 'Validación'], loc='upper right')


y_pred = model.predict(x_test)

cnf_matrix = confusion_matrix(y_test, y_pred.round())
funciones.plot_confusion_matrix(cnf_matrix, classes=['Normal','Evento Extremo'], normalize=True ,title='Caudales')
