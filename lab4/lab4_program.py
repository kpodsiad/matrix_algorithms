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
    col[0] = 0
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
    NNZ = len(data)
    # traverse over diagonal
    for k in range(N):
        col_offset = colptr[k]
        next_col_offset = colptr[k + 1]  # next column starting idx
        while row[col_offset] != k:  # find element from diagonal
            col_offset += 1
        akk = data[col_offset]

        # get info about non-zero values in column under diagonal position
        data_to_delete = data[col_offset + 1:next_col_offset]
        row_to_delete = row[col_offset + 1:next_col_offset]

        if len(data_to_delete) == 0:  # nothing will be deleted, there is no reason to do anything more
            continue

        # traverse over values in k row after diagonal one
        for j in range(k + 1, N):
            # find non zero values in k row
            j_col_idx = colptr[j]
            # while row[j] is still above diagonal and j still belongs to k column, not k+1
            while row[j_col_idx] < k and j_col_idx < colptr[j + 1]:
                j_col_idx += 1
            akj = data[j_col_idx]
            # if there is no value in k row or it's zero there nothing to do
            if row[j_col_idx] > k or akj == 0:
                continue

            for d_row, d_data in zip(row_to_delete, data_to_delete):
                i = j_col_idx + 1
                # traverse until find proper index
                while row[i] < d_row and i < colptr[j + 1]:
                    i += 1
                # there are 2 cases now
                # first current row is greater than deleted row - that means it's "0 - x" situation
                # and a new non zero must be created
                if row[i] != d_row or i == colptr[j + 1]:
                    data = np.concatenate(
                        (data[:i, ], np.zeros(1), data[i:, ]))
                    row = np.concatenate(
                        (row[:i, ], np.array([d_row]), row[i:, ]))
                    colptr += (colptr >= i) * 1
                # second - current row is equal deleted row and existing value must be corrected
                # but this is common logic for both cases
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

