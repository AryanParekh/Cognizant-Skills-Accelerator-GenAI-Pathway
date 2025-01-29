import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Part 1.1: Choose and Load the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# Part 1.2: Preprocess (Load, Normalize)
iris_data = pd.read_csv(url, names=columns)
scaler = StandardScaler()
iris_data_scaled = scaler.fit_transform(iris_data.iloc[:, :-1])

# Part 1.3: Implementing Clustering

# K-means
distortions = []
K = range(1, 11)
for k in K:
    kmeanModel = KMeans(n_clusters=k,random_state=43)
    kmeanModel.fit(iris_data_scaled)
    distortions.append(kmeanModel.inertia_)

plt.figure(figsize=(10,6))
plt.plot(K, distortions, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

# Hierarchical Clustering
Z = linkage(iris_data_scaled, 'ward')
plt.figure(figsize=(10, 6))
dendrogram(Z, truncate_mode='lastp', p=12, leaf_rotation=45., leaf_font_size=15., show_contracted=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Cluster size')
plt.ylabel('Distance')
plt.show()

# Part 2.4: Visualize Clusters
kmeanModel = KMeans(n_clusters=3,random_state=43)
kmeanModel.fit(iris_data_scaled)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(iris_data_scaled)
principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])

plt.figure(figsize=(10,6))
plt.scatter(principalDf['principal component 1'], principalDf['principal component 2'], c=kmeanModel.labels_, cmap='rainbow')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA of Iris Dataset Clusters')
plt.show()

# Part 3.1: Cluster Labels
iris_data['cluster'] = kmeanModel.labels_
crosstab = pd.crosstab(iris_data['cluster'], iris_data['class'])
print(crosstab)

# Part 3.2: Evaluation Metrics
silhouette_avg = silhouette_score(iris_data_scaled, iris_data['cluster'])
ari_score = adjusted_rand_score(iris_data['class'], iris_data['cluster'])
print(f"Silhouette Score: {silhouette_avg:.2f}")
print(f"Adjusted Rand Index: {ari_score:.2f}")