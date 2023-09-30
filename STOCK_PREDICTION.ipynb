#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[39]:


import pandas_datareader as pdr
key="2ea3b82f7d0222c24d6efc54c3a64e2c55365d9e"


# In[40]:


df = pdr.get_data_tiingo('AAPL', api_key="2ea3b82f7d0222c24d6efc54c3a64e2c55365d9e")


# In[41]:


df.to_csv('AAPL.csv')


# In[42]:


import pandas as pd


# In[43]:


df=pd.read_csv('AAPL.csv')


# In[44]:


df.head()


# In[45]:


df.tail()


# In[46]:


df1=df.reset_index()['close']


# In[47]:


df1


# In[48]:


import matplotlib.pyplot as plt
plt.plot(df1)


# In[50]:


import numpy as np


# In[51]:


df1


# In[52]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[53]:


print(df1)


# In[54]:


##splitting dataset into train and test split
training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[55]:


training_size,test_size


# In[56]:


len(train_data),len(test_data)


# In[57]:


train_data


# In[58]:


import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)


# In[59]:


time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


# In[60]:


print(X_train.shape), print(y_train.shape)


# In[61]:


print(X_test.shape), print(ytest.shape)


# In[62]:


# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# In[63]:


### Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[64]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[65]:


model.summary()


# In[66]:


model.summary()


# In[67]:


model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)


# In[69]:


### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[70]:


##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[71]:


### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[72]:


### Test Data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))


# In[73]:


### Plotting 
# shift train predictions for plotting
look_back=100
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[74]:


len(test_data)


# In[75]:


x_input=test_data[340:].reshape(1,-1)
x_input.shape


# In[76]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[77]:


temp_input


# In[78]:


from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)


# In[79]:


day_new=np.arange(1,101)
day_pred=np.arange(101,131)


# In[80]:


import matplotlib.pyplot as plt


# In[81]:


len(df1)


# In[82]:


plt.plot(day_new,scaler.inverse_transform(df1[1156:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))


# In[83]:


df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[1200:])


# In[84]:


df3=scaler.inverse_transform(df3).tolist()


# In[85]:


plt.plot(df3)


# In[ ]:




