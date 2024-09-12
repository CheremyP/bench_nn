from tensorflow import random, sparse, matmul
from numpy import random as np_random

# Create a benchmark for sparse and dense matrix multiplication
def bench_nn_tensorflow():
    """
    This function benchmarks the performance of sparse and dense matrix multiplication using tensorflow.
    """
    
    # Create a dense matrix
    dense = random.normal((1000, 1000))
    # Create a sparse matrix
    indices = np_random.randint(0, 1000, (2, 1000))
    values = np_random.randn(1000)
    sparse_matrix = sparse.SparseTensor(indices=indices, values=values, dense_shape=(1000, 1000))

    # Benchmark dense matrix multiplication
    for _ in range(100):
        matmul(dense, dense)

    # Benchmark sparse matrix multiplication
    for _ in range(100):
        sparse.sparse_dense_matmul(sparse_matrix, sparse_matrix)

if __name__ == '__main__':
    bench_nn_tensorflow()
