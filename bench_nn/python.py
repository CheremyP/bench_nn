import random

# Create a benchmark for sparse and dense matrix multiplication
def bench_nn_python():
    # Create a dense matrix
    dense = [[random.random() for _ in range(1000)] for _ in range(1000)]
    # Create a sparse matrix
    sparse = {(random.randint(0, 999), random.randint(0, 999)): random.random() for _ in range(1000)}

    # Benchmark dense matrix multiplication
    for _ in range(100):
        dense_result = [[sum(a * b for a, b in zip(row, col)) for col in zip(*dense)] for row in dense]

    # Benchmark sparse matrix multiplication
    for _ in range(100):
        sparse_result = [[sum(sparse.get((i, k), 0) * dense[k][j] for k in range(1000)) for j in range(1000)] for i in range(1000)]

if __name__ == '__main__':
    bench_nn_python()
