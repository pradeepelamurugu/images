# Commented out IPython magic to ensure Python compatibility.
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
# %matplotlib inline

data=pd.read_csv('adult.csv')

data.head(10)

data.columns

plt.scatter(data.Age,data['Income'])
plt.xlabel('Age')
plt.ylabel('Income')

km=KMeans(n_clusters=3)

y_predicted = km.fit_predict(data[['Age','Income']])
y_predicted

y_predicted

data['cluster']=y_predicted
data.head()

data['cluster']=y_predicted
data.head(10)

km.cluster_centers_

df1 = data[data.cluster==0]
df2 = data[data.cluster==1]
df3 = data[data.cluster==2]
plt.scatter(df1.Age,df1['Income'],color='green')
plt.scatter(df2.Age,df2['Income'],color='red')
plt.scatter(df3.Age,df3['Income'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('Age')
plt.ylabel('Income')
plt.legend()

data.columns

sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(data[['Age','Income']])
    sse.append(km.inertia_)

plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)

