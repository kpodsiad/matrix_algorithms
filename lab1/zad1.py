import numpy as np
import os
import time

m = 600
n = 600
k = 600

A = np.arange(1, m * k + 1).reshape(m, k).astype(np.float)
B = np.arange(1, k * n + 1).reshape(k, n).astype(np.float)


def multiply_matrix(A, B):
    m, k1 = A.shape
    k2, n = B.shape
    assert k1 == k2
    C = np.zeros(m * n).reshape(m, n)

    for p in range(k1):
        for j in range(n):
            a = A[:, p]
            b = B[p, j]
            C[:, j] += a * b
    return C


os.system('clear')

start = time.time()
C = multiply_matrix(A, B)
end = time.time()
print(f'computation takes {round(end - start, 2)} seconds')
