from itertools import combinations
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import davies_bouldin_score, silhouette_score

def p_y_x_params(data, k, phis, means, covars):
    probs = phis*[multivariate_normal(means[j], covars[j]).pdf(data) for j in range(k)]
    return probs/np.sum(probs, axis=0)

def cluster(data, k, tol=.00001):
    # Initialize parameters randomly
    phis = np.random.rand(k, 1)
    means = data[np.random.choice(data.shape[0], k, replace=False)]
    covars = np.array([np.cov(data.T, bias=True)]*k)
    while True:
        old_means = means.copy()
        # E-step
        w = p_y_x_params(data, k, phis, means, covars)
        # M-step
        phis = np.mean(w, axis=1, keepdims=True)
        means = (w @ data)/np.sum(w, axis=1, keepdims=True)
        for j in range(k):
            resid = data - means[j]
            covars[j] = ((w[[j]] * resid.T) @ resid) / np.sum(w[j])
        # Finish if not much movement
        max_change = np.amax(np.abs(old_means - means))
        if max_change < tol:
            break
    return phis, means, covars

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
            phis, means, covars = cluster(d[:, 1].reshape(-1, 1), k, tol=.01)
            c = np.argmax([multivariate_normal(means[j], covars[j]).pdf(d[:, 1])*phis[j] for j in range(k)], axis=0)
            all_means.append(np.sort(means, axis=0))
            all_n_right = all_n_pairs = 0
            for j in range(k):
                c_k = d[c == j]
                n_pairs = c_k.shape[0]*(c_k.shape[0] - 1)/2
                pairs = combinations(c_k[:, 0], 2)
                n_right = sum(1 for a, b in pairs if a == b)
                all_n_pairs += n_pairs
                all_n_right += n_right
            all_stdvs.append(np.mean(covars))
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
            phis, means, covars = cluster(d[:, 1].reshape(-1, 1), k, tol=.01)
            c = np.argmax([multivariate_normal(means[j], covars[j]).pdf(d[:, 1])*phis[j] for j in range(k)], axis=0)
            all_means.append(np.sort(means, axis=0))
            all_n_right = all_n_pairs = 0
            for j in range(k):
                c_k = d[c == j]
                n_pairs = c_k.shape[0]*(c_k.shape[0] - 1)/2
                pairs = combinations(c_k[:, 0], 2)
                n_right = sum(1 for a, b in pairs if a == b)
                all_n_pairs += n_pairs
                all_n_right += n_right
            all_stdvs.append(np.mean(covars))
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
        phis, means, covars = cluster(d[:, 1].reshape(-1, 1), k, tol=.01)
        c = np.argmax([multivariate_normal(means[j], covars[j]).pdf(d[:, 1])*phis[j] for j in range(k)], axis=0)
        all_means.append(np.sort(means, axis=0))
        all_n_right = all_n_pairs = 0
        for j in range(k):
            c_k = d[c == j]
            n_pairs = c_k.shape[0]*(c_k.shape[0] - 1)/2
            pairs = combinations(c_k[:, 0], 2)
            n_right = sum(1 for a, b in pairs if a == b)
            all_n_pairs += n_pairs
            all_n_right += n_right
        all_stdvs.append(np.mean(covars))
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
            phis, means, covars = cluster(d[:, 1].reshape(-1, 1), k, tol=.01)
            c = np.argmax([multivariate_normal(means[j], covars[j]).pdf(d[:, 1])*phis[j] for j in range(k)], axis=0)
            all_means.append(np.sort(means, axis=0))
            all_n_right = all_n_pairs = 0
            for j in range(k):
                c_k = d[c == j]
                n_pairs = c_k.shape[0]*(c_k.shape[0] - 1)/2
                pairs = combinations(c_k[:, 0], 2)
                n_right = sum(1 for a, b in pairs if a == b)
                all_n_pairs += n_pairs
                all_n_right += n_right
            all_stdvs.append(np.mean(covars))
            all_n_w.append(all_n_right/all_n_pairs)
        print("Means:", np.mean(all_means, axis=0))
        print("Stdvs:", np.mean(all_stdvs))
        print("Fracs:", np.mean(all_n_w))

def cluster_urban(k):
    data = np.genfromtxt("urban/urbanGB.txt", delimiter=",")[:1000]

    phis, means, covars = cluster(data, k)
    c = np.argmax([multivariate_normal(means[j], covars[j]).pdf(data)*phis[j] for j in range(k)], axis=0)

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

if __name__ == "__main__":
    cluster_experiment_1()
    # cluster_experiment_2()
    # cluster_experiment_3()
    # cluster_experiment_4()
    # cluster_urban(k=10)
