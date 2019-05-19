import pandas as pd
bikes = pd.read_csv('bikes.csv')
bikes.head()
from matplotlib import pyplot as plt
plt.figure(figsize=(8,6))
plt.plot(bikes['temperature'], bikes['count'], 'o')
plt.xlabel('temperature')
plt.ylabel('bikes')
plt.show()
from sklearn.tree import DecisionTreeRegressor
import numpy as np
regressor = DecisionTreeRegressor(max_depth=2)
regressor.fit(np.array([bikes['temperature']]).T, bikes['count'])
xx = np.array([np.linspace(-5, 40, 100)]).T
plt.figure(figsize=(8,6))
plt.plot(bikes['temperature'], bikes['count'], 'o', label='observation')
plt.plot(xx, regressor.predict(xx), linewidth=4, alpha=.7, label='prediction')
plt.xlabel('temperature')
plt.ylabel('bikes')
plt.legend()
plt.show()
import pydot
!dot -Tpng tree.dot > tree.png
from IPython.display import Image
Image(filename='tree.png')