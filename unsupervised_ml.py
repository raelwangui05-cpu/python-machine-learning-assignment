from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#data preparation(monthly spending and visits)
x=[
    [1000,2],
    [1500,3],
    [2000,2],
    [5000,6],
    [6000,7],
    [5500,5],
    [12000,12],
    [15000,14],
    [13000,11],
    [3000,3]
]

#use standardscaler to scale the spending and visits features

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

#create KMeans model with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

#train the model
kmeans.fit(x_scaled)

#Display the cluster assigned to each customer
print("Cluster assignments for each customer:")
for i, label in enumerate(kmeans.labels_):
    print(f"Customer {i+1}: Cluster {label}")

#Display the total number of customers in each cluster
from collections import Counter
cluster_counts = Counter(kmeans.labels_)
print("\nTotal number of customers in each cluster:")
for cluster, count in cluster_counts.items():
    print(f"Cluster {cluster}: {count} customers")

#visualisation
import matplotlib.pyplot as plt

plt.scatter(x_scaled[:, 0], x_scaled[:, 1], c=kmeans.labels_, cmap='viridis')
plt.xlabel('Scaled Spending')
plt.ylabel('Scaled Visits')
plt.title('Customer Segments')
plt.show()
