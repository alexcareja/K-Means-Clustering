import pandas
import numpy
import matplotlib.pyplot as plot

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

N = 2
N_COMPONENTS = 2
MAX_ITER = 1000
N_CLUSTERS = 3
H = 0.005 	# Decrease for better quality
LINEWIDTHS = 3
ZORDER = 10


# Read Credit Card Data
ccd = pandas.read_csv("CC GENERAL.csv")
ccd = ccd.drop("CUST_ID", axis = 1)
ccd.fillna(method = 'ffill', inplace = True)
ccd.head(N)

normalized_data = normalize(StandardScaler().fit_transform(ccd))
normalized_data = pandas.DataFrame(normalized_data)

# Applying PCA
pca = PCA(N_COMPONENTS)
X = pca.fit_transform(normalized_data)
X = pandas.DataFrame(X)
X.columns = ['P1', 'P2']
X.head(N)

# Applying Elbow Criterion
ssd = {}
for k in range(1,10):
	kmeans = KMeans(k, max_iter = MAX_ITER).fit(X)
	ssd[k] = kmeans.inertia_	# Sum of squared distances of samples to their closest cluster center.

plot.figure()
plot.plot(list(ssd.keys()), list(ssd.values()))
plot.xlabel("Number of clusters")
plot.ylabel("Sum of squared distances")
plot.show()

# Since Elbow criterion is often not very reliable, we are going to implement the 
# Sillhouette Coeficient Method
scores = []
for k in range(2,8):
	scores.append(
		silhouette_score(X, KMeans(k).fit_predict(X)))
plot.bar([x for x in range(2,8)], scores)
plot.xlabel("Number of clusters")
plot.ylabel("Sillhouette score")
plot.show()

# Visualise the Clusters for N_CLUSTERS = 3
kmeans = KMeans(N_CLUSTERS)
kmeans.fit(X)
plot.scatter(X['P1'], X['P2'], c = KMeans(N_CLUSTERS).fit_predict(X))
plot.show()

# Visualise Clusters & Centroids
x_min = X['P1'].min() - 1
x_max = X['P1'].max() + 1
y_min = X['P2'].min() - 1
y_max = X['P2'].max() + 1

XX, YY = numpy.meshgrid(numpy.arange(x_min, x_max, H), numpy.arange(y_min, y_max, H))
Z = kmeans.predict(numpy.array(list(zip(XX.ravel(), YY.ravel()))))
Z = Z.reshape(XX.shape)
plot.figure(1)
plot.clf()
plot.imshow(Z, interpolation = 'nearest', 
	extent = (XX.min(), XX.max(), YY.min(), YY.max()),
	aspect = 'auto', origin = 'lower')
plot.plot(X['P1'], X['P2'], 'k.', markersize = 1)
centroids = kmeans.cluster_centers_
plot.scatter(centroids[:, 0], centroids[:, 1], marker = 'o', 
	s = 10, color ='w', linewidths = LINEWIDTHS, zorder = ZORDER)
plot.xlim(x_min, x_max)
plot.ylim(y_min, y_max)
plot.title("Representation of Clusters\nCredit card dataset")
plot.show()