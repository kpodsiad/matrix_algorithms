import unittest
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from lab4_program import csc_gauss_elimination, coo_gauss_elimination


def matrix_to_coo(matrix):
    a = coo_matrix(matrix)
    return a.col, a.row, a.data


def csc_to_matrix(N, colptr, row, data):
    return csc_matrix((data, row, colptr), shape=(N, N)).toarray()


class CSCTestCase(unittest.TestCase):

    def test_diagonal_matrix(self):
        matrix = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])
        result = coo_gauss_elimination(3, matrix_to_coo(matrix))
        matrix2 = csc_to_matrix(3, *result)
        self.assertTrue(np.allclose(matrix, matrix2))

    def test_dense_ones_matrix(self):
        matrix = np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]])
        result = coo_gauss_elimination(3, matrix_to_coo(matrix))
        matrix2 = csc_to_matrix(3, *result)
        self.assertTrue(np.allclose(matrix, matrix2))

    def test_matrix_that_has_value_in_every_column(self):
        matrix = np.array([[1, 1, 0],
                           [0, 1, 1],
                           [0, 0, 1]])
        result = coo_gauss_elimination(3, matrix_to_coo(matrix))
        matrix2 = csc_to_matrix(3, *result)
        self.assertTrue(np.allclose(matrix, matrix2))
