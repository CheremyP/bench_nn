import torch 

# Create a bench mark for sparse and dense matrix multiplication
def bench_nn_torch():
    # Create a dense matrix
    dense = torch.randn(1000, 1000)
    # Create a sparse matrix
    sparse = torch.sparse_coo_tensor(indices=torch.randint(0, 1000, (2, 1000)), values=torch.randn(1000), size=(1000, 1000))

    # Benchmark dense matrix multiplication
    for _ in range(100):
        torch.matmul(dense, dense)

    # Benchmark sparse matrix multiplication
    for _ in range(100):
        torch.sparse.mm(sparse, dense)

if __name__ == '__main__':
    bench_nn_torch()
