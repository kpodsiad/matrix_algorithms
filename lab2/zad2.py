import numpy as np
import math

N = 3
A = np.random.rand(N, N)

print(A)

for k in range(N):
    j = k + np.argmax(A[k, k:])

    assert not math.isclose(A[k, j], 0, abs_tol=1e-8), "matrix is singular"

    if j != k:
        A[[k, j]] = A[[j, k]]  # swap k and j rows

    A[k + 1:N, k] /= A[k, k]  # divide diagonal column
    for j in range(k + 1, N):  # for others column in k row
        A[k + 1:N, j] -= A[k + 1:N, k] * A[k, j]  # j column -= k column * A_kj

print(A)
