from itertools import combinations
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

def cluster_experiment_1():
    data = []
    data.append(np.genfromtxt("out/File100.tsv", delimiter="\t"))
    data.append(np.genfromtxt("out/File101.tsv", delimiter="\t"))
    data.append(np.genfromtxt("out/File102.tsv", delimiter="\t"))
    data.append(np.genfromtxt("out/File103.tsv", delimiter="\t"))
    data.append(np.genfromtxt("out/File104.tsv", delimiter="\t"))
    data.append(np.genfromtxt("out/File105.tsv", delimiter="\t"))
    data.append(np.genfromtxt("out/File106.tsv", delimiter="\t"))
    data.append(np.genfromtxt("out/File107.tsv", delimiter="\t"))
    data.append(np.genfromtxt("out/File108.tsv", delimiter="\t"))
    data.append(np.genfromtxt("out/File109.tsv", delimiter="\t"))
    data.append(np.genfromtxt("out/File110.tsv", delimiter="\t"))
    data.append(np.genfromtxt("out/File111.tsv", delimiter="\t"))
    for l in range(len(data)):
        print(l)
        all_means = []
        all_stdvs = []
        all_n_w = []
        for i in range(50):
            d = data[l]
            k = len(np.unique(d[:, 0]))
            c, means = cluster(d[:, 1].reshape(-1, 1), k, tol=.01)
            all_means.append(np.sort(means, axis=0))
            all_n_right = all_n_pairs = 0
            stdvs = []
            for j in range(k):
                c_k = d[c == j]
                n_pairs = c_k.shape[0]*(c_k.shape[0] - 1)/2
                pairs = combinations(c_k[:, 0], 2)
                n_right = sum(1 for a, b in pairs if a == b)
                all_n_pairs += n_pairs
                all_n_right += n_right
                stdvs.append(d[c == j, 1].std())
            all_stdvs.append(np.mean(stdvs))
            all_n_w.append(all_n_right/all_n_pairs)
        print("Means:", np.mean(all_means, axis=0))
        print("Stdvs:", np.mean(all_stdvs))
        print("Fracs:", np.mean(all_n_w))

def cluster_experiment_2():
    data = []
    data.append(np.genfromtxt("out/File200.tsv", delimiter="\t"))
    data.append(np.genfromtxt("out/File201.tsv", delimiter="\t"))
    data.append(np.genfromtxt("out/File202.tsv", delimiter="\t"))
    for l in range(len(data)):
        print(l)
        all_means = []
        all_stdvs = []
        all_n_w = []
        for i in range(50):
            d = data[l]
            k = 5
            c, means = cluster(d[:, 1].reshape(-1, 1), k, tol=.01)
            all_means.append(np.sort(means, axis=0))
            all_n_right = all_n_pairs = 0
            stdvs = []
            for j in range(k):
                c_k = d[c == j]
                n_pairs = c_k.shape[0]*(c_k.shape[0] - 1)/2
                pairs = combinations(c_k[:, 0], 2)
                n_right = sum(1 for a, b in pairs if a == b)
                all_n_pairs += n_pairs
                all_n_right += n_right
                stdvs.append(d[c == j, 1].std())
            all_stdvs.append(np.mean(stdvs))
            all_n_w.append(all_n_right/all_n_pairs)
        print("Means:", np.mean(all_means, axis=0))
        print("Stdvs:", np.mean(all_stdvs))
        print("Fracs:", np.mean(all_n_w))

def cluster_experiment_3():
    data = np.genfromtxt("out/File300.tsv", delimiter="\t")
    all_means = []
    all_stdvs = []
    all_n_w = []
    for i in range(50):
        d = data
        k = 5
        c, means = cluster(d[:, 1].reshape(-1, 1), k, tol=.01)
        all_means.append(np.sort(means, axis=0))
        all_n_right = all_n_pairs = 0
        stdvs = []
        for j in range(k):
            c_k = d[c == j]
            n_pairs = c_k.shape[0]*(c_k.shape[0] - 1)/2
            pairs = combinations(c_k[:, 0], 2)
            n_right = sum(1 for a, b in pairs if a == b)
            all_n_pairs += n_pairs
            all_n_right += n_right
            stdvs.append(d[c == j, 1].std())
        all_stdvs.append(np.mean(stdvs))
        all_n_w.append(all_n_right/all_n_pairs)
    print("Means:", np.mean(all_means, axis=0))
    print("Stdvs:", np.mean(all_stdvs))
    print("Fracs:", np.mean(all_n_w))

def cluster_experiment_4():
    data = []
    data.append(np.genfromtxt("out/File400.tsv", delimiter="\t"))
    data.append(np.genfromtxt("out/File401.tsv", delimiter="\t"))
    data.append(np.genfromtxt("out/File402.tsv", delimiter="\t")[:10000])
    for l in range(len(data)):
        print(l)
        all_means = []
        all_stdvs = []
        all_n_w = []
        for i in range(50):
            d = data[l]
            k = 5
            c, means = cluster(d[:, 1].reshape(-1, 1), k, tol=.01)
            all_means.append(np.sort(means, axis=0))
            all_n_right = all_n_pairs = 0
            stdvs = []
            for j in range(k):
                c_k = d[c == j]
                n_pairs = c_k.shape[0]*(c_k.shape[0] - 1)/2
                pairs = combinations(c_k[:, 0], 2)
                n_right = sum(1 for a, b in pairs if a == b)
                all_n_pairs += n_pairs
                all_n_right += n_right
                stdvs.append(d[c == j, 1].std())
            all_stdvs.append(np.mean(stdvs))
            all_n_w.append(all_n_right/all_n_pairs)
        print("Means:", np.mean(all_means, axis=0))
        print("Stdvs:", np.mean(all_stdvs))
        print("Fracs:", np.mean(all_n_w))

def cluster_urban(k):
    data = np.genfromtxt("urban/urbanGB.txt", delimiter=",")[:1000]

    c, means = cluster(data, k)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data[:, 0], data[:, 1], c=c, s=1)
    plt.xlim([-7, 2])
    plt.ylim([50, 59])
    ax.set_aspect("equal")
    plt.show()

    print("SSE  v:", np.sum([np.sum((data - means[j])**2) for j in range(k)]))
    print("Sil  ^:", silhouette_score(data, c))
    print("DB   v:", davies_bouldin_score(data, c))

def cluster_image(k):
    img = Image.open("images/Great_Wave_kInf.jpg")
    data = np.asarray(img)
    data = data.transpose(1, 0, 2).reshape(-1, 3)

    c, means = cluster(data, k)

    pix = img.load()
    for i in range(data.shape[0]):
        x = i // img.size[1]
        y = i % img.size[1]
        pix[x, y] = tuple(np.around(means[c[i]]).astype(int))
    # img.save("images/Great_Wave_k10.jpg")
    img.show()

if __name__ == "__main__":
    cluster_experiment_1()
    # cluster_experiment_2()
    # cluster_experiment_3()
    # cluster_experiment_4()
    # cluster_urban(k=10)
    # cluster_image(k=3)
