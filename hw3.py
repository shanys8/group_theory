import numpy as np
import pandas as pd
import math
from numpy import linalg as LA
from scipy import linalg
import time
from scipy.stats import ortho_group


def question1():
    return


def assert_special_orthogonal(x, n):
    assert (np.allclose(np.dot(x, x.T), np.identity(n), rtol=1.e-2, atol=1.e-2))
    assert(np.isclose(np.fabs(linalg.det(x)), 1))


def generate_special_orthogonal_sample(n):
    x = ortho_group.rvs(n)
    assert_special_orthogonal(x, n)
    return x


def generate_samples(n, N):
    samples = np.empty((N, n, n), float)
    i = 0
    while i < N:
        samples[i] = generate_special_orthogonal_sample(n)
        i += 1
    # return samples
    return np.array([
        [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
        [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
        [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]
    ])



def generate_Hij(samples, i, j):
    return np.dot(samples[i], samples[j].T)


def generate_row(samples, N, n, i):
    row = np.empty((n, 0), float)
    j = 0
    while j < N:
        x = np.identity(n) if (i == j) else generate_Hij(samples, i, j)
        row = np.append(row, x, axis=1)
        j += 1
    return row


def build_M(samples, n, N):
    rows = np.empty((0, N*n), float)

    i = 0
    while i < N:
        row = generate_row(samples, N, n, i)
        rows = np.append(rows, row, axis=0)
        i += 1
    return rows


def build_V(M, n):
    w, v = linalg.eigh(M)
    return v[:, -n:]


def getBi(V, i, n):
    return V[n*i:(n*i+3), :]


def calculate_O_opt(samples, V, n, N):
    i = 0
    A = np.zeros((n, n), float)
    while i < N:
        A += np.dot(samples[i].T, getBi(V, i, n))
        i += 1
    U, S, Vh = linalg.svd(A)
    return np.dot(Vh, U.T)


def validate_results(samples, V, n, N):
    Q_opt = calculate_O_opt(samples, V, n, N)
    error = 0
    i = 0
    # approx. samples should be similar to original samples
    while i < N:
        approx_sample = np.dot(getBi(V, i, n), Q_opt)
        error += math.pow(LA.norm(samples[i] - approx_sample , ord='fro'), 2)
        print('approx_sample')
        print(approx_sample)
        print('samples[i]')
        print(samples[i])
        assert (np.allclose(approx_sample, samples[i], rtol=1.e-2, atol=1.e-2))
        i += 1

    # error for optimal Q should be zero
    assert(np.isclose(error, 0, rtol=1.e-2, atol=1.e-2))

def main():
    n = 3
    N = 3
    samples = generate_samples(n, N)

    # print('samples')
    # print(samples)

    M = build_M(samples, n, N)

    print('M')
    print(M)

    V = build_V(M, n)

    validate_results(samples, V, n, N)



    return 0


if __name__ == "__main__":
    main()
