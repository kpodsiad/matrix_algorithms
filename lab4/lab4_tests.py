import unittest

import numpy as np
import scipy.sparse.linalg as la
from scipy.sparse import coo_matrix, csc_matrix

from lab4_program import coo_gauss_elimination


def matrix_to_coo(matrix):
    a = coo_matrix(matrix)
    return a.col, a.row, a.data


def coo_to_matrix(N, data, row, col):
    return coo_matrix((data, (row, col)), shape=(N, N)).toarray()


class CSCTestCase(unittest.TestCase):

    def test_diagonal_matrix(self):
        m = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
        coo_result = coo_gauss_elimination(len(m[0]), matrix_to_coo(m))
        result = coo_to_matrix(len(m[0]), *coo_result)
        self.assertTrue(np.allclose(m, result))

    def test_matrix_that_has_value_in_every_column(self):
        m = np.array([[1, 1, 0],
                      [0, 1, 1],
                      [0, 0, 1]])
        csc_result = coo_gauss_elimination(len(m[0]), matrix_to_coo(m))
        result = coo_to_matrix(len(m[0]), *csc_result)
        self.assertTrue(np.allclose(m, result))

    def test_dense_ones_matrix(self):
        m = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 11]])
        expected_result = np.array([[1, 2, 3],
                                    [0, -3, -6],
                                    [0, 0, 2]])
        csc_result = coo_gauss_elimination(len(m[0]), matrix_to_coo(m))
        result = coo_to_matrix(len(m[0]), *csc_result)
        self.assertTrue(np.allclose(expected_result, result))

    def test_matrix_1(self):
        m = np.array([[1, 2, 1],
                      [0, 28, 6],
                      [7, 0, 11]])
        expected_result = np.array([[1, 2, 1],
                                    [0, 28, 6],
                                    [0, 0, 7]])
        csc_result = coo_gauss_elimination(len(m[0]), matrix_to_coo(m))
        result = coo_to_matrix(len(m[0]), *csc_result)
        self.assertTrue(np.allclose(expected_result, result))

    def test_matrix_2(self):
        m = np.array([[1, -8, 0],
                      [0, 8, -3],
                      [1, 0, 1]])
        expected_result = np.array([[1, -8, 0],
                                    [0, 8, -3],
                                    [0, 0, 4]])
        csc_result = coo_gauss_elimination(len(m[0]), matrix_to_coo(m))
        result = coo_to_matrix(len(m[0]), *csc_result)
        self.assertTrue(np.allclose(expected_result, result))

    def test_matrix_3(self):
        m = np.array([[1, -9, 0, 0],
                      [4, 4, 0, 0],
                      [3, 0, 1, 0],
                      [0, 0, 4, -1]])
        expected_result = np.array([[1, -9, 0, 0],
                                    [0, 40, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, -1]])
        csc_result = coo_gauss_elimination(len(m[0]), matrix_to_coo(m))
        result = coo_to_matrix(len(m[0]), *csc_result)
        self.assertTrue(np.allclose(expected_result, result))

    def test_matrix_4(self):
        m = np.array([[2, 0, 0, 2],
                      [0, 1, 0, 0],
                      [-6, 0, 1, 0],
                      [-4, 0, 0, 1]])
        expected_result = np.array([[2, 0, 0, 2],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 6],
                                    [0, 0, 0, 5]])
        csc_result = coo_gauss_elimination(len(m[0]), matrix_to_coo(m))
        result = coo_to_matrix(len(m[0]), *csc_result)
        self.assertTrue(np.allclose(expected_result, result))

    def test_matrix_5(self):
        m = np.array([[1, -4, 5, 0],
                      [6, 1, 0, 0],
                      [-5, 0, 1, 0],
                      [0, 3, 0, 1]])
        expected_result = np.array([[1, -4, 5, 0],
                                    [0, 25, -30, 0],
                                    [0, 0, 2, 0],
                                    [0, 0, 0, 1]])
        csc_result = coo_gauss_elimination(len(m[0]), matrix_to_coo(m))
        result = coo_to_matrix(len(m[0]), *csc_result)
        self.assertTrue(np.allclose(expected_result, result))

    def test_matrix_6(self):
        m = np.array([[8, 8, 0, 0, 0],
                      [0, 1, 3, 0, 8],
                      [-8, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0],
                      [0, 4, 0, 0, -8]])
        expected_result = np.array([[8, 8, 0, 0, 0],
                                    [0, 1, 3, 0, 8],
                                    [0, 0, -23, 1, -64],
                                    [0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, -152 / 23]])
        csc_result = coo_gauss_elimination(len(m[0]), matrix_to_coo(m))
        result = coo_to_matrix(len(m[0]), *csc_result)
        self.assertTrue(np.allclose(expected_result, result))

    def test_matrix_7(self):
        m = np.array([[1, 0, 0, 8],
                      [0, 1, 5, -3],
                      [0, 8, 1, 0],
                      [0, 9, 0, 1]])
        expected_result = np.array([[1, 0, 0, 8],
                                    [0, 1, 5, -3],
                                    [0, 0, -39, 24],
                                    [0, 0, 0, 4 / 13]])
        csc = csc_matrix(m)
        B = la.splu(csc, permc_spec='NATURAL', diag_pivot_thresh=0,
                    options={"SymmetricMode": True})
        B.U.sort_indices()
        csc_result = coo_gauss_elimination(len(m[0]), matrix_to_coo(m))
        result = coo_to_matrix(len(m[0]), *csc_result)
        self.assertTrue(np.allclose(expected_result, result))
