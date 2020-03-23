from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np

datas = open('air_bnb.tsv').read().splitlines()
ATTR = datas[0].split()

c = np.array([(0, 0)])
for _ in datas[1:]:
	PRICE = _.split('\t')[ATTR.index('price')]
	REVIEW = _.split('\t')[ATTR.index('reviews_per_month')]

	if not PRICE or not REVIEW:
		continue

	c = np.vstack((c, (
			float(PRICE),
			float(REVIEW)
	)))

k = KMeans(n_clusters=2, init='k-means++', max_iter=len(c), n_init=1, random_state=0)
k.fit_predict(c)

plt.scatter(c[:, 0], c[:, 1], c=k.labels_)
plt.scatter(k.cluster_centers_[:, 0], k.cluster_centers_[:, 1], s=len(c), c='red')

plt.show()