from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.metrics import davies_bouldin_score, silhouette_score

def cluster(data, k, tol=.00001):
    # Pick k random points as initial means
    means = data[np.random.choice(data.shape[0], k, replace=False)]
    while True:
        old_means = means.copy()
        # Cluster the data
        dists = cdist(data, means)
        c = np.array([np.argmin(d) for d in dists])
        # Move the means
        for j in range(k):
            means[j] = np.mean(data[c == j], axis=0)
        # Finish if not much movement
        max_change = np.amax(np.abs(old_means - means))
        if max_change < tol:
            break
    return c, means

def main():
    # img = Image.open("images/Great_Wave_kInf.jpg")
    # data = np.asarray(img)
    # data = data.transpose(1, 0, 2).reshape(-1, 3)

    data = np.genfromtxt("data/unlabeled.csv", delimiter=",", skip_header=1)

    k = 2
    c, means = cluster(data, k)

    # pix = img.load()
    # for i in range(data.shape[0]):
    #     x = i // img.size[1]
    #     y = i % img.size[1]
    #     pix[x, y] = tuple(np.around(means[c[i]]).astype(int))
    # # img.save("images/Great_Wave_k10.jpg")
    # img.show()

    # plt.scatter(data[:, 0], data[:, 1], c=c)
    # plt.show()

    print(np.sum([np.sum((data-means[j])**2) for j in range(k)]))
    print(silhouette_score(data, c))
    print(davies_bouldin_score(data, c))

if __name__ == "__main__":
    main()
