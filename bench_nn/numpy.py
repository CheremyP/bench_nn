from numpy import random, sparse, dot, matmul

def bench_nn_numpy():
    """
    This function benchmarks the performance of sparse and dense matrix multiplication using numpy.
    """

    # Create a dense matrix
    dense = random.randn(1000, 1000)
    # Create a sparse matrix
    indices = random.randint(0, 1000, (2, 1000))
    values = random.randn(1000)
    sparse_matrix = sparse.coo_matrix((values, indices), shape=(1000, 1000))

    # Benchmark dense matrix multiplication
    for _ in range(100):
        matmul(dense, dense)

    # Benchmark sparse with sparse matrix multiplication
    for _ in range(100):
        dot(sparse_matrix, sparse_matrix)
        
if __name__ == '__main__':
    bench_nn_numpy()
