#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 16:13:20 2018

@author: paulo
"""
## PROYECTO PAULO GRANADOS
#Librerías
import csv 
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#Extracción de la data desde archivo csv
data = []
with open('creditcard.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader: 
        print(', '.join(row))
        data.append(row)
        
#Preparación de la data para ingresarla a la red
def extraer_data(data):
    dataset = np.zeros((len(data)-1,31))
    for i in range(len(data)-1):
        aux = data[i+1]
        aux2 = aux[0]
        aux3 = aux2.split(',')
        for j in range(len(aux3)):
            if aux3[j] ==  ('"0"'):
                dataset[i,j] = 0
                
            elif aux3[j] == ('"1"'):
                dataset[i,j] = 1
                
            else:
                dataset[i,j] = float(aux3[j])
                
    dataset2 = dataset[:,1:] 
    dataset_final = np.delete(dataset2,28,1) #matriz final con toda la información
    
    data_fraude = np.where(dataset_final[0:,28] == 1)
    fraudes = dataset_final[np.array(data_fraude[0])] #matriz con toda la data fraudulenta
    
    data_save = np.where(dataset_final[0:,28] == 0)
    seguras = dataset_final[np.array(data_save[0])] #matriz con toda la data segura
    
    fraude_train = fraudes[0:int(len(fraudes)*0.6),:] #60% de entrenamiento
    seguras_train = seguras[0:int(len(seguras)*0.6),:]
    fraude_test = fraudes[int(len(fraudes)*0.6):,:] #40% de test
    seguras_test = seguras[int(len(seguras)*0.6):,:]    
    train = np.concatenate((seguras_train,fraude_train)) #toda la data de entrenamiento (seguras y fraudes)
    
    return train, seguras_test, fraude_test

train, seguras_test, fraude_test = extraer_data(data) # se retorna data de entrenamiento y de test

X = train[:,0:28] # variables
Y = train[:,28] # etiquetas
fraude_X = fraude_test[:,0:28] # variables
seguras_X = seguras_test[:,0:28] # variables

# modelo de la red neuronal
model = Sequential()
model.add(Dense(28, input_dim=28, init= 'uniform' , activation= 'relu' )) # capa de entrada y primera capa oculta
model.add(Dense(4, init= 'uniform' , activation= 'relu' )) # segunda capa oculta
model.add(Dense(1, init= 'uniform' , activation= 'sigmoid' )) # capa de salida

model.compile(loss= 'binary_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ]) # Se compila el modelo
# se entrena el modelo
model.fit(X, Y, nb_epoch=100, batch_size=10) # se entrena el modelo

#model.save_weights("model14_2.h5") # se guarda el modelo

model.load_weights("model28_4.h5") # se carga el mejor modelo obtenido

# RESULTADOS
res_fraudes = []
for i in range(len(fraude_X)):
    
    prueba = fraude_X[i]
    prueba = np.reshape(prueba, (1,-1))
    
    predict = model.predict(prueba)
    res_fraudes.append(predict)

score_fraudes = 0
for i in range(len(res_fraudes)):
    if res_fraudes[i] >= 0.5:
        score_fraudes = score_fraudes +1 # aciertos

score_fraudes_per = (score_fraudes * 100)/len(res_fraudes) # porcentaje de aciertos

res_seguras = []
for i in range(len(seguras_X)):
    prueba = seguras_X[i]
    prueba = np.reshape(prueba, (1,-1))
    
    predict = model.predict(prueba)
    res_seguras.append(predict)    

score_seguras = 0
for i in range(len(res_seguras)):
    if res_seguras[i] < 0.5:
        score_seguras = score_seguras +1 # aciertos

score_seguras_per = (score_seguras*100)/len(res_seguras) # porcentaje de aciertos

# Ejemplo de uso
# Se debe tomar una fila de la data segura o de fraude de test, aplicar un reshape y luego predecir con model.predict

prueba2 = fraude_X[80]
prueba2 = np.reshape(prueba2, (1,-1))

predict_ejemplo = model.predict(prueba2)

if predict_ejemplo[0] > 0.5:
    print("Transacción fraudulenta")
else:
    print("Transacción segura")





