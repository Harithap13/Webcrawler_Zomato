import seaborn as sns
sns.set()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
data=pd.read_csv('zom22.csv')

plt.scatter(data['cost_for_two'],data['rating'])
plt.xlabel('Cost for 2')
plt.ylabel('Rating')
plt.show()


x=data[['cost_for_two','rating']].copy()
###########################################
kmeans=KMeans(3)
kmeans.fit(x)
clusters=x.copy()
clusters['cluster_pred']=kmeans.fit_predict(x)

plt.scatter(clusters['cost_for_two'],clusters['rating'],c=clusters['cluster_pred'],cmap='rainbow')
plt.xlabel('Cost for 2')
plt.ylabel('Rating')
plt.show()
##############################################
#FOR KMEANS
from sklearn import preprocessing
x_scaled=preprocessing.scale(x)
wcss=[]
for i in range(1, 30):
    kmeans = KMeans(i)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,30),wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
####################################################
kmeans_new=KMeans(4)
kmeans.fit(x_scaled)
cluster_new=x.copy()
cluster_new['cluster_pred']=kmeans_new.fit_predict(x_scaled)

plt.scatter(cluster_new['cost_for_two'],cluster_new['rating'],c=cluster_new['cluster_pred'],cmap='rainbow')
plt.xlabel('Cost for 2')
plt.ylabel('Rating')
plt.show()

