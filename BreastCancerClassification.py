# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 14:29:11 2023

@author: sachin kumar
"""

#importing the dependencies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split

#data collection and preprocessing
breast_cancer_dataset=sklearn.datasets.load_breast_cancer()

print(breast_cancer_dataset)

#loading the data to data frame
data_frame=pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)

print(data_frame.head())

data_frame['label']=breast_cancer_dataset.target

print(data_frame.tail())

print(data_frame.info())

print(data_frame.isnull().sum())

print(data_frame.describe())

#checking the distribution of target variable
print(data_frame['label'].value_counts())

#1->Benign
#0->malignant

print(data_frame.groupby('label').mean())

#seperating the features and target
X=data_frame.drop(columns='label',axis=1)
Y=data_frame['label']

print(X)
print(Y)


#splitting the dataset into training and test data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)

print(X.shape,X_train.shape,X_test.shape)

#standardizing the data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_std=scaler.fit_transform(X_train)
X_test_std=scaler.transform(X_test)


#Building the neural network
#importing tensorflow and keras

import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras


#setting up the layers of neural network
model=keras.Sequential([
    keras.layers.Flatten(input_shape=(30,)),
    keras.layers.Dense(20,activation='relu'),
    keras.layers.Dense(2,activation='sigmoid')])

#compiling the neural network
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#training the neural network
history=model.fit(X_train_std,Y_train,validation_split=0.1,epochs=10)

#visualising acccuracy and 
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')

plt.legend(['training data','validation data'],loc='lower right')


#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])

#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['training data','validation data'],loc='center right')


#Accuracy of the model on the test data
loss,accuracy=model.evaluate(X_test_std,Y_test)
print(accuracy)

print(X_test_std.shape)
print(X_test_std[0])


Y_pred=model.predict(X_test_std)
print(Y_pred.shape)
print(Y_pred[0])

print(X_test_std)
print(Y_pred)

#converting the prediction probability to class labels
Y_pred_labels=[np.argmax(i) for i in Y_pred]
print(Y_pred_labels)

#Building the predictive system
input_data=(17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189)
#change the input data to numpy array
input_data_as_numpy_array=np.asarray(input_data)

#reshape the numpy array as predicting for one data point
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

#standardizing the input data
input_data_std=scaler.transform(input_data_reshaped)

prediction=model.predict(input_data_std)
print(prediction)

prediction_label=[np.argmax(prediction)]
print(prediction_label)

if(prediction_label[0]==0):
    print("The breast cancer is malignant")
else:
    print("The Breast cancer is benign")
