import tensorflow as tf
import numpy as np

# Create a benchmark for sparse and dense matrix multiplication
def bench_nn_tensorflow():
    # Create a dense matrix
    dense = tf.random.normal((1000, 1000))
    # Create a sparse matrix
    indices = np.random.randint(0, 1000, (2, 1000))
    values = np.random.randn(1000)
    sparse = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=(1000, 1000))

    # Benchmark dense matrix multiplication
    for _ in range(100):
        tf.matmul(dense, dense)

    # Benchmark sparse matrix multiplication
    for _ in range(100):
        tf.sparse.sparse_dense_matmul(sparse, dense)

if __name__ == '__main__':
    bench_nn_tensorflow()
