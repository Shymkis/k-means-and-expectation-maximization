from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import davies_bouldin_score, silhouette_score

def p_y_x_params(data, k, phis, means, covars):
    probs = phis*[multivariate_normal(means[j], covars[j]).pdf(data) for j in range(k)]
    return probs/np.sum(probs, axis=0)

def run_em(data, k, tol=.00001):
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

def main():
    data = np.genfromtxt("data/unlabeled.csv", delimiter=",", skip_header=1)

    k = 2
    phis, means, covars = run_em(data, k)
    c = np.argmax([multivariate_normal(means[j], covars[j]).pdf(data)*phis[j] for j in range(k)], axis=0)

    # plt.scatter(data[:,0], data[:,1], c=c)
    # plt.show()

    print(np.sum([np.sum((data-means[j])**2) for j in range(k)]))
    print(silhouette_score(data, c))
    print(davies_bouldin_score(data, c))

if __name__ == "__main__":
    main()
