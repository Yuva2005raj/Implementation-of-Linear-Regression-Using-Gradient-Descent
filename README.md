# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard python libraries for Gradient design.

2. Introduce the variables needed to execute the function.

3. Use function for the representation of the graph.

4. Using for loop apply the concept using the formulae.

5. Execute the program and plot the graph.

6. Predict and execute the values for the given conditions.

## Program:
```

Program to implement the linear regression using gradient descent.
Developed by: YUVARAJ B
RegisterNumber: 212222230182 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("/content/ex1.txt",header=None)

print("Profit prediction graph:")
plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,1000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  """
  Take in a numpy array X,y,theta and generate the cost function in a linear regression model
  """
  m=len(y) #length of the training data
  h=X.dot(theta) #hypothesis
  square_err=(h-y)**2

  return 1/(2*m) * np.sum(square_err) #returning
  
  data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

print("Compute cost value:")
computeCost(X,y,theta) #Call the function

def gradientDescent(X,y,theta,alpha,num_iters):
  """
  Take in numpy array X,y and theta and update theta by taking number with learning rate of alpha

  return theta and the list of the cost of theta during each iteration
  """
  m=len(y)
  J_history=[]

  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions -y))
    descent=alpha * 1/m * error
    theta-=descent
    J_history.append(computeCost(X,y,theta))

  return theta,J_history  
  
theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) value:")
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

print("Cost function using Gradient Descent graph:")
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

print("Profit prediction graph:")
plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("profit ($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
  """
  Take in numpy array of x and theta and return the predicted value of y based on theta
  """

  predictions= np.dot(theta.transpose(),x)

  return predictions[0]
  
predict1=predict(np.array([1,3.5]),theta)*10000
print("Profit for the population 35,000:")
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("Profit for the population 70,000:")
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))


```

## Output:

## profit predicted graph:

![268232479-673c0ffa-ae7f-4702-8e11-78393b4d8a2d](https://github.com/Yuva2005raj/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118343998/ee51db96-9183-4060-aa41-32933e93998c)

# compute cost value:

![268232556-fe50f74a-2f2c-456f-a2b8-44b58ca77e76](https://github.com/Yuva2005raj/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118343998/f6304ee3-e49d-4c01-a0c2-5710ca194d5e)

# h(x)value:
![268232581-d306795e-1849-4a28-b815-b086ede8eabf](https://github.com/Yuva2005raj/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118343998/4cccd7c5-3c11-435c-86ef-333e905e97d9)


## cost function using gradient descent graph:

![268232608-a76d12a7-ba8f-45aa-8490-e9264c013503](https://github.com/Yuva2005raj/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118343998/4780f453-bf0e-4c7d-abf8-43d3a9685209)

## profit prediction graph:

![268232650-b9ca5e73-3d15-4d12-87b0-e830a3adc08f](https://github.com/Yuva2005raj/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118343998/d2db09ff-fed9-4853-b158-20610e5d9eac)

# profit for the population 35,000:

![268232709-74ac0a17-259b-4ce6-9320-32c514484fcc](https://github.com/Yuva2005raj/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118343998/de1a1a26-22b0-4a64-b182-f280959572c7)

# profit for the population 70,000:

![268232737-7b57e4cd-a671-49c9-9b52-ca83128eaf47](https://github.com/Yuva2005raj/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118343998/3e2fc000-65a1-4be7-9f81-238329e55fe7)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
