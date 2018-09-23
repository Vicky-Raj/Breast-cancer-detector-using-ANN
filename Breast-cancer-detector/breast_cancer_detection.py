"""
Created on Thu Aug 30 17:11:09 2018

@author: vignesh
"""
#Importing necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers

#Preprocessing
data = pd.read_csv('data.csv')
X = data.iloc[:,2:-1].values
y = data.iloc[:,1].values
encoder = LabelEncoder()
y = encoder.fit_transform(y)
scaler = StandardScaler()
X = scaler.fit_transform(X)
#Splitting
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.1)
#Defining the ANN
model = Sequential()
model.add(Dense(32,input_shape=(30,),activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(16,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(2,activation='softmax',kernel_regularizer=regularizers.l2(0.01)))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#Fitting to the Dataset
model.fit(x_train,y_train,epochs=100,batch_size=10,verbose=1)
#Evalutaion metrics
y_pred = model.predict_classes(x_test)
metrics = model.evaluate(x_test,y_test)
conf_matrix = confusion_matrix(y_test,y_pred)
print("Loss: {}\nAccuracy: {}".format(metrics[0],metrics[1]))
#Saving the model
model.save('breast_cancer_detection.h5')