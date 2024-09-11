import jax

def bench_nn_jax():
    # Create a dense matrix
    dense = jax.random.normal(jax.random.PRNGKey(0), (1000, 1000))
    # Create a sparse matrix
    indices = jax.random.randint(jax.random.PRNGKey(0), 0, 1000, (2, 1000))
    values = jax.random.normal(jax.random.PRNGKey(0), (1000,))
    sparse = jax.scipy.sparse.coo_matrix((values, indices), shape=(1000, 1000))

    # Benchmark dense matrix multiplication
    for _ in range(100):
        jax.numpy.matmul(dense, dense)

    # Benchmark sparse matrix multiplication
    for _ in range(100):
        jax.numpy.dot(sparse, dense)

if __name__ == '__main__':
    bench_nn_jax()