import numpy as np
import math
import random

from numpy import linalg as LA
from scipy import linalg
from scipy.stats import ortho_group
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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
    return samples
    # return np.array([
    #     [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
    #     [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
    #     [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]
    # ])


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
    dim_M = M.shape[0]-1
    w, v = linalg.eigh(M, eigvals=(dim_M - n + 1, dim_M))
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


def round_block(B):
    U, S, Vh = linalg.svd(B)
    return LA.det(np.dot(U, Vh)) * np.dot(U, Vh)


def validate_results(samples, V, n, N):
    Q_opt = calculate_O_opt(samples, V, n, N)
    error = 0
    i = 0
    # approx. samples should be similar to original samples
    while i < N:
        approx_sample = np.dot(round_block(getBi(V, i, n)), Q_opt)
        error += math.pow(LA.norm(samples[i] - approx_sample , ord='fro'), 2)
        # q1 -  validate approx sample is close enough to the original sample
        assert (np.allclose(approx_sample, samples[i], rtol=1.e-2, atol=1.e-2))
        i += 1

    # q2 -  error for optimal Q should be zero
    assert(np.isclose(error, 0, rtol=1.e-2, atol=1.e-2))


def synchronization_problem():
    n = 3
    N = 3
    samples = generate_samples(n, N)
    M = build_M(samples, n, N)
    V = build_V(M, n)
    validate_results(samples, V, n, N)


# generate M matrix
def sample_so_matrix():
    n = 3
    # generate ramdom angle between 0 and 2pi
    alpha = 2 * math.pi * random.random()
    # build rotation matrix in alpha angle - z axis remains the same
    R = np.array([[math.cos(alpha), math.sin(alpha), 0], [-math.sin(alpha), math.cos(alpha), 0], [0, 0, 1]])
    # generate v - random point on the unit sphere
    (x, y) = (random.uniform(0, 1), random.uniform(0, 1))
    v = np.array([math.cos(2 * math.pi * x) * math.sqrt(y), math.sin(2 * math.pi * x) * math.sqrt(y), math.sqrt(1-y)])
    # reflection matrix H
    H = np.identity(n) - 2 * np.dot(v[:, np.newaxis], v[np.newaxis, :])
    # rotation matrix M
    M = (-1) * np.dot(H, R)
    return M


# generate 1000 samples from some x in R3 (in my case x=[0,0,1]) multiplied by M
def validate_sample_so_matrix():
    n = 3
    x = np.array([0, 0, 1])
    results = np.empty((n, 0), float)
    i = 0
    while i < 1000:
        M = sample_so_matrix()
        result = np.dot(M, x)[:, np.newaxis]
        results = np.append(results, result, axis=1)
        i += 1
    visualize_results(results)


def visualize_results(results):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = results[0]
    y = results[1]
    z = results[2]

    ax.scatter(x, y, z, c='r', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


def main():

    synchronization_problem()  # q1 and q2
    validate_sample_so_matrix()  # q3
    return 0


if __name__ == "__main__":
    main()
