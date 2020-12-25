from scipy.sparse import csc_matrix, random


def csc_to_coordinate_format(values, rows, colptr):
    """
    arguments:
    values - Array that contains the non-zero elements of A.
    implicit argument NNZ = len(values)

    rows - Element i of the integer array rows is the number of the row in A that contains the i-th value in the values array

    colptr - Array that contains row offsets i.e values from col i start at colptr[i] index in values array.
    colptr[i] = colptr[i-1] + nnz(col_i-1) and colptr[0] = 0
    Note that the length of colptr is NNZ + 1 where the last element in colptr is the number of nonzeros in sparse matrix
    
    """
    NNZ = len(values)
    col = []
    col_idx = 0
    next_col_ptr_idx = colptr[col_idx+1]
    # loop through values <==> traverse the columns one by one
    for val_idx in range(NNZ):
        while val_idx >= next_col_ptr_idx: # go to the next column and skip empty ones
            col_idx += 1
            next_col_ptr_idx = colptr[col_idx+1]
        col.append(col_idx)
    return values, rows, col


N = 30
density = 0.2
# use scipy to generate random sparse matrix in csc format
A_csc = csc_matrix(random(N,N, density=density))
# transform csc format to coo
# data,row,col = csc_to_coordinate_format(A_csc.data, A_csc.indices, A_csc.indptr)

# verify implementation using scipy
# A_coo = coo_matrix(A_csc)
# assert np.array_equal(A_coo.col, np.array(col))
# assert np.array_equal(A_coo.row, np.array(row))
# assert np.array_equal(A_coo.data, np.array(data))


# trivial example from lecture
N = 3
rows = [0,2,1,1,2]
values = [1,4,2,3,5]
colptr = [0,2,3,5]
# prints values, rows, columns
for v in csc_to_coordinate_format(values, rows, colptr):
    print(v)
