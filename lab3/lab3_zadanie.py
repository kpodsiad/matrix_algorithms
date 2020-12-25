import numpy as np
from scipy.sparse import csr_matrix, random


def multiply_csr_by_vector(N, values, cols, rowptr, vector):
    """
    arguments:
    N - matrix size

    values - Array that contains the non-zero elements of A.

    cols - Element i of the integer array cols is the number of the col in A that contains the i-th value in the values array

    rowptr - Array that contains row offsets i.e values from col i start at rowptr[i] index in values array.
    rowptr[i] = rowptr[i-1] + nnz(col_i-1) and rowptr[0] = 0
    Note that the length of rowptr is NNZ + 1 where the last element in rowptr is the number of nonzeros in sparse matrix

    vector - vector to mult with matrix 
    
    """
    Y = [0.0 for _ in range(N)]  # initialize result vector
    row_idx = 0
    next_row_idx = rowptr[row_idx + 1]
    # loop through values <==> traverse the rows one by one
    for idx, value in enumerate(values):
        while idx >= next_row_idx:  # go to the next row and skip empty ones
            row_idx += 1
            next_row_idx = rowptr[row_idx + 1]
        from_vector = vector[cols[idx]]  # take nth element from vector where n is value's column
        Y[row_idx] += value * from_vector  # add mult result to

    return Y


# use scipy to algorithm verification
N = 50
density = 0.1

A_csr = csr_matrix(random(N, N, density=density))
vector = np.ones(N)
# result from own implementation
result = multiply_csr_by_vector(N, A_csr.data, A_csr.indices, A_csr.indptr, vector)

assert np.array_equal(A_csr * vector, result)
