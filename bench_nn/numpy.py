import jax.numpy as jnp
from jax import random
from jax.scipy import sparse
import time

# Create a benchmark for sparse and dense matrix multiplication
def bench_nn_jax():
    # Create a random key for reproducibility
    key = random.PRNGKey(0)

    # Create a dense matrix
    dense = random.normal(key, (1000, 1000))

    # Create a sparse matrix
    key, subkey = random.split(key)
    indices = random.randint(subkey, (2, 1000), 0, 1000)
    values = random.normal(subkey, (1000,))
    sparse_matrix = sparse.coo_matrix((values, (indices[0], indices[1])), shape=(1000, 1000))

    # Benchmark dense matrix multiplication
    start_time = time.time()
    for _ in range(100):
        jnp.matmul(dense, dense)
    dense_time = time.time() - start_time
    print(f'Dense matrix multiplication time: {dense_time:.4f} seconds')

    # Benchmark sparse matrix multiplication
    start_time = time.time()
    for _ in range(100):
        sparse_matrix.dot(dense)
    sparse_time = time.time() - start_time
    print(f'Sparse matrix multiplication time: {sparse_time:.4f} seconds')

if __name__ == '__main__':
    bench_nn_jax()