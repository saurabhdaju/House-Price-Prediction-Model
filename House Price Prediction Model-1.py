#!/usr/bin/env python
# coding: utf-8

# #  House Price Prediction Model  
# ### (using linear regression in one variable)

# ### We are going to use the relation between "house area" and "house price" to predict the price of houses.
# 

# In[2]:


#importing libraries
import numpy as np


# In[3]:


house_area = np.array([2.2, 0.4, 4, 0.8])    # area in 1000sqft.

house_price = np.array([1, 0.5, 2, 0.8])     # price in Crores

#input size n
n = len(house_area)


# In[4]:


alpha = 0.1 

# Learning Rate is the amount of jump we make everytime to come closer to minima (of cost function).
# it ranges between 0 and 1 
# smaller the value more is the accuracy and running time
# higher the value lesser the accuracy and running time


# In[5]:


def derivate(x, y, m, c):
    
    derivative1 = 0;
    derivative2 = 0;
    
    for i in range(len(x)):
        derivative1 += (m * x[i] + c - y[i]) * x[i]
        derivative2 +=  m * x[i] + c - y[i]
        
    return derivative1, derivative2
    


# In[6]:


def minJ(m, c, alpha, x, y, n):
    dev1, dev2 = derivate(x, y, m, c)
    
    iterations = 100
    
    while(iterations >= 0):
        temp1 = m - (alpha * dev1) / n
        temp2 = c - (alpha * dev2) / n
        
        m, c = temp1, temp2
        dev1, dev2 = derivate(x, y, m, c)
        
        iterations -= 1
    
    return m, c
        


# In[7]:


print("INPUT DATA")
print()
print("FEATURE    TARGET")
for i in range(n):
    print(f"{house_area[i]}        {house_price[i]}")


# In[8]:


# considering initial values of 'm' and 'c'

m = 2
c = 0.4

print(f"Assumed values\n m = {m}\n c = {c}")


# In[9]:


# calling the function to compute correct values of 'm' and 'c'
m, c = minJ(m, c, alpha, house_area, house_price, n)

print(f"Correct values\n m = {m}\n c = {c}")


# ## Therefore our model is ready to use

# In[10]:


area = float(input("Enter the house area "))

price = m * area + c
print('The expected price should be %.4f Crores' %price)


# In[ ]:




