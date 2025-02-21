import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
from sklearn.decomposition import PCA

df = pd.read_csv("student.csv")
X = df[["studytime",'freetime','goout',"absences","G3","failures","G1","G2"]] 

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calculate inertia for different values of k
inertia = []
k_values = range(2, 10)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.show()

# Calculate silhouette scores for different values of k
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(score)

# Plot the Silhouette Scores
plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Optimal k")
plt.show()

# Apply K-means clustering
k = 4  # Optimal number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Reduce data to 2D using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

# Plot the clusters
plt.figure(figsize=(10, 6))
# sns.scatterplot(x="absences", y="G3", hue="Cluster", data=df, palette="viridis", s=100)
sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster", data=df, palette="viridis", s=100)
plt.title("2D PCA Visualization of Clusters (K=4)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
# plt.title("2D Visualization of Clusters")
# plt.xlabel("Absences")
# plt.ylabel("G3")
plt.legend(title="Cluster")
plt.show()


from mpl_toolkits.mplot3d import Axes3D

# Reduce data to 3D using PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Add PCA components to the DataFrame
df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]
df["PCA3"] = X_pca[:, 2]

# Plot the clusters in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(df["PCA1"], df["PCA2"], df["PCA3"], c=df["Cluster"], cmap="viridis", s=100)

# Add labels and title
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_zlabel("PCA Component 3")
plt.title("3D PCA Visualization of Clusters (k=4)")

# Add a color bar
legend = ax.legend(*scatter.legend_elements(), title="Cluster")
ax.add_artist(legend)

plt.show()
