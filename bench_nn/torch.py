from torch import randn, randint, sparse, matmul

# Create a benchmark for sparse and dense matrix multiplication
def bench_nn_torch():
    """
    This function benchmarks the performance of sparse and dense matrix multiplication using torch.
    """
    # Create a dense matrix
    dense = randn(1000, 1000)
    # Create a sparse matrix
    sparse_matrix = sparse.FloatTensor(indices=randint(0, 1000, (2, 1000)), values=randn(1000), size=(1000, 1000))

    # Benchmark dense matrix multiplication
    for _ in range(100):
        matmul(dense, dense)

    # Benchmark sparse matrix multiplication
    for _ in range(100):
        sparse.mm(sparse_matrix, sparse_matrix)


if __name__ == '__main__':
    bench_nn_torch()
