import numpy as np

# Create a benchmark for sparse and dense matrix multiplication
def bench_nn_numpy():
    # Create a dense matrix
    dense = np.random.randn(1000, 1000)
    # Create a sparse matrix
    indices = np.random.randint(0, 1000, (2, 1000))
    values = np.random.randn(1000)
    sparse = np.sparse.coo_matrix((values, indices), shape=(1000, 1000))

    # Benchmark dense matrix multiplication
    for _ in range(100):
        np.matmul(dense, dense)

    # Benchmark sparse matrix multiplication
    for _ in range(100):
        np.dot(sparse, dense)

if __name__ == '__main__':
    bench_nn_numpy()
