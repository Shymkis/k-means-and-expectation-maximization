import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time

def cluster(data, k, tol=.00001):
    n_exs, n_dims = data.shape
    r = np.ptp(data, axis=0)
    m = np.amin(data, axis=0)
    mean = np.random.rand(k, n_dims)*r + m
    c = np.zeros((n_exs, 1))
    while True:
        old_mean = mean.copy()
        for i in range(n_exs):
            a = [0]*k
            for j in range(k):
                a[j] = np.linalg.norm(data[i] - mean[j])
            c[i] = np.argmin(a)
        for j in range(k):
            d = np.sum(c == j)
            if d > 0:
                n = np.sum((c == j)*data, axis=0)
                mean[j] = n / d
        max_change = np.amax(np.abs(old_mean - mean))
        if max_change < tol:
            break
    return c, mean

def main():
    img = Image.open("images/Mona_Lisa.jpg")
    data = np.asarray(img)
    data = data.transpose(1, 0, 2).reshape(-1, 3)

    start = time.time()
    c, mean = cluster(data, k=3)
    end = time.time()

    pix = img.load()
    for i in range(data.shape[0]):
        x = i // img.size[1]
        y = i % img.size[1]
        pix[x, y] = tuple(mean[int(c[i])].astype(int))

    img.show()

    time_elapsed = end - start
    print("Time elapsed:", round(time_elapsed, 2))

if __name__ == "__main__":
    main()
