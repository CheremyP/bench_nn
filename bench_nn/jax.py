import time
import jax as jnp
from jax import random, matmul
from scipy.sparse import coo_matrix

# Constants
MATRIX_SIZE = (1000, 1000)
NUM_ITERATIONS = 100
RANDOM_SEED = 0

def bench_nn_jax():
    """
    Benchmarks the performance of sparse and dense matrix multiplication using JAX.
    """
    key = random.PRNGKey(RANDOM_SEED)

    # Create a dense matrix
    dense_matrix = random.normal(key, MATRIX_SIZE)

    # Create a sparse matrix
    key, subkey = random.split(key)
    indices = random.randint(subkey, (2, 1000), 0, 1000)
    values = random.normal(subkey, (1000,))
    sparse_matrix = coo_matrix((values, (indices[0], indices[1])), shape=MATRIX_SIZE)

    # Benchmark dense matrix multiplication
    start_time = time.time()
    for _ in range(NUM_ITERATIONS):
        matmul(dense_matrix, dense_matrix)
    dense_time = time.time() - start_time
    print(f'Dense matrix multiplication time: {dense_time:.4f} seconds')

    # Benchmark sparse matrix multiplication
    start_time = time.time()
    for _ in range(NUM_ITERATIONS):
        sparse_matrix.dot(dense_matrix)
    sparse_time = time.time() - start_time
    print(f'Sparse matrix multiplication time: {sparse_time:.4f} seconds')

if __name__ == '__main__':
    bench_nn_jax()