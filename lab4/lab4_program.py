import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, random, identity
import os
from quicksort import quicksort
import scipy.sparse.linalg as la

os.system('clear')


def coo_to_csc(N, col, row, data):
    quicksort((col, row, data), 0, len(data))
    k = 0
    # merge same cords
    for i in range(1, len(data)):
        if col[k] != col[i] or row[k] != row[i]:
            k += 1
            col[k], row[k], data[k] = col[i], row[i], data[i]
        else:
            data[k] += data[i]

    data, row = data[:k + 1], row[:k + 1]
    # create colptr
    col.resize(len(data) + 1, refcheck=False)
    k = 0
    c = col[0]
    for i in range(1, len(data)):
        c1 = col[i]
        if c1 != c:
            k += 1
            col[k] = i
        c = c1
    col[k + 1] = len(data)
    col = col[:k + 2]

    return col, row, data


def coo_gauss_elimination(N, coo):
    col, row_coo, data_coo = coo
    colptr, row, data = coo_to_csc(N, col, row_coo, data_coo)
    return csc_gauss_elimination(N, colptr, row, data)


def csc_gauss_elimination(N, colptr, row, data):
    # traverse over diagonal
    for k in range(N - 1):
        col_offset = colptr[k]
        next_col_offset = colptr[k + 1]  # next column starting idx
        while row[col_offset] != k:  # find element from diagonal
            col_offset += 1
        akk = data[col_offset]

        # get info about non-zero values in column under diagonal position
        data_to_delete = data[col_offset + 1:next_col_offset]
        row_to_delete = row[col_offset + 1:next_col_offset]

        if len(data_to_delete) == 0:  # nothing will be deleted, here's nothing to do anything more
            continue

        # traverse over values in k row after diagonal one
        for j in range(k + 1, N):
            # find non zero values in k row

            j_column_data = data[colptr[j]:colptr[j + 1]]

            j_rows = row[colptr[j]:colptr[j + 1]]

            # k row not in j_rows means that A[k,j] = 0 and there's nothing to do anything more
            if k not in j_rows:
                continue

            akj = j_column_data[np.where(j_rows == k)[0][0]]

            for d_row, d_data in zip(row_to_delete, data_to_delete):
                if d_row not in row[colptr[j]:colptr[j + 1]]:  # "0 - x" situation
                    res_idx = np.where(row[colptr[j]:colptr[j + 1]] > d_row)[0]

                    i = (colptr[j] + res_idx[0]) if len(res_idx) != 0 else colptr[j + 1]
                    data = np.concatenate((data[:i, ], np.zeros(1), data[i:, ]))
                    row = np.concatenate((row[:i, ], np.array([d_row]), row[i:, ]))
                    colptr += (colptr >= i) * 1
                else:
                    i = colptr[j] + np.where(row[colptr[j]:colptr[j + 1]] == d_row)[0][0]

                data[i] -= akj * (d_data / akk)
        # finally delete values from row
        data[col_offset + 1:next_col_offset] = 0

    k = 0
    for i in range(len(data)):
        if data[i] != 0:
            row[k] = row[i]
            data[k] = data[i]
            k += 1
        else:
            colptr -= colptr > k * 1

    row, data = row[:k], data[:k]

    return colptr, row, data

# N = 3
# density = 0.8
# A = csc_matrix(random(N, N, density=density, dtype="int"))

# c,r,d = coo_to_csc(3, np.array([1,0,0,2,0,1,2]), np.array([2,0,1,0,0,1,2]), np.array([3,3,4,2,-2,1,1]))

# col  = np.array([0,2,4,6])
# row  = np.array([0,1,1,2,0,2])
# data = np.array([1,4,3,1,2,1])
# D = np.array([[1,9,3,5,11,7],
#               [0,4,8,2,10,6]])

# print(csc_gauss_elimination(N, c, r, d))

# N = 5
# density = 0.3
# coo = coo_matrix(random(N, N, density=density, dtype="int") + identity(N))
# coo = coo_matrix(np.array(
# ))
# csc = coo.tocsc()
# c, r, d = coo_to_csc(N, coo.col.copy(), coo.row.copy(), coo.data.copy())
#
# assert np.array_equal(c, csc.indptr), "colptr is not equal"
# assert np.array_equal(r, csc.indices), "row is not equal"
# assert np.array_equal(d, csc.data), "data is not equal"
#
# B = la.splu(csc, permc_spec='NATURAL', diag_pivot_thresh=0, options={"SymmetricMode": True})
# B.U.sort_indices()
# col, row, data = csc_gauss_elimination(N, c.copy(), r.copy(), d.copy())
#
# assert np.array_equal(col, B.U.indptr), "colptr is not equal #2"
# assert np.array_equal(row, B.U.indices), "row is not equal #2"
# print(csc.toarray())
# print(col, B.U.indptr)
# assert np.allclose(data, B.U.data), "data is not equal #2"
#
# i = 100
# while i < 50:
#     clear()
#     coo = coo_matrix(random(N, N, density=density, dtype="int") + identity(N))
#     csc = coo.tocsc()
#     print(csc.toarray())
#     c, r, d = coo_to_csc(N, coo.col.copy(), coo.row.copy(), coo.data.copy())
#     if not np.array_equal(c, csc.indptr):
#         print("colptr is not equal")
#         i = 201
#     if not np.array_equal(r, csc.indices):
#         print("row is not equal")
#         i = 201
#     if not np.array_equal(d, csc.data):
#         print("data is not equal")
#         i = 201
#     #
#     B = la.splu(csc, permc_spec='NATURAL', diag_pivot_thresh=0,
#                 options={"SymmetricMode": True})
#     B.U.sort_indices()
#
#     col, row, data = csc_gauss_elimination(N, c.copy(), r.copy(), d.copy())
#     #
#     if not np.array_equal(col, B.U.indptr):
#         print("colptr is not equal #2", i)
#         i = 201
#     if not np.array_equal(row, B.U.indices):
#         print("row is not equal #2", i)
#         i = 201
#     if not np.allclose(data, B.U.data):
#         print("data is not equal #2", i)
#         i = 201
#     #
#     i += 1
# #
