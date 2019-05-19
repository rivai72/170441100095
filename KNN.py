import numpy as np 
import pandas as pd
from subprocess import check_output
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('Drugs.csv')
y=df.iloc[:,-1]
X=df.iloc[:, :-1]
X.head()


gender_encoder = LabelEncoder()
#Male=1, Female=0
y = gender_encoder.fit_transform(y)
y
#Standardize features by removing the mean and scaling to unit variance
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics

knn=KNeighborsClassifier() 
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
print('Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred))
k_range=list(range(1,len(X_train)))
acc_score=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k) 
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    acc_score.append(metrics.accuracy_score(y_test,y_pred))
import matplotlib.pyplot as plt

k_values=list(range(1,len(X_train)))
plt.plot(k_values,acc_score)
plt.xlabel('Value of k for knn')
plt.ylabel('Accuracy')
import operator
index, value = max(enumerate(acc_score), key=operator.itemgetter(1))
index
value