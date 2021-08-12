# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 08:43:06 2021

@author: FARZAN
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
#%%
dataset = pd.read_csv("LR.csv")
x= dataset.iloc[:,0:2].values
y= dataset.iloc[:,2].values
#%%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state= 0)
#%%
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
#%%
from sklearn.ensemble import RandomForestClassifier
classifer = RandomForestClassifier(n_estimators= 200, criterion= "entropy", random_state=0)
classifer.fit(x_train,y_train)
print(classifer.score(x_test,y_test)*100)
y_pred = classifer.predict(x_test)
#%%
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
#%%
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred)*100)
#%%
#Visualising the Training set results
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(x1, x2, classifer.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.50, cmap = ListedColormap(("red", "green")))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

for i, j in enumerate (np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(("red", "green"))(i), label = j)  
plt.title("DecisionTreeClassifier (Training set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show() 

#%%
from matplotlib.colors import ListedColormap
x_set, y_set = x_test,y_test
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(x1, x2, classifer.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.50, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('DecisionTreeClassifier(Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
