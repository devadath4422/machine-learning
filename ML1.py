#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np  #importing function
import pandas as pd
import matplotlib.pyplot as plt
def ComputeCost(X,y,theta): #defining a function computecost
    n=len(y) #length
    h=X.dot(theta)
    square_err=(h-y)**2
    return 1/(2*m)*np.sum(square_err)
def gradientDescent (X,y,theta,alpha,num_iters): #defining a function as gradientDescent
    n=len(y)
    J_history=[]
    for i in  range(num_iters):
        h=X.dot(theta)
        error=np.dot(X.transpose(),(h-y))
        descent=alpha*(1/m)*error
        theta-=descent
        J_history.append(ComputeCost(X,y,theta))
    return theta,J_history #returning theeta
data=pd.read_csv('ex1data1.txt',header=None)
data_n=data.values
m=len(data_n[:,0]) #length of the data
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
ComputeCost(X,y,theta)
theta,J_history=gradientDescent(X,y,theta,0.01,1500)
theta[0,0]
plt.scatter(data[0],data[1]) #ploting
plt.plot(X,theta[1,0]*X+theta[0,0]) #ploting
plt.show #ploting


# In[ ]:




