import cupy as cp
from datetime import datetime
import csv


def generate_large_system_cupy(size, seed):
    cp.random.seed(seed)
    A_cp = cp.random.rand(size, size)
    B_cp = cp.random.rand(size)
    return A_cp, B_cp


def test_cupy_solve(size, seed):
    A, B = generate_large_system_cupy(size, seed)
    start_time = datetime.now()
    X = cp.linalg.solve(A, B)
    cp.cuda.Stream.null.synchronize()  # Ensure GPU operations are completed
    end_time = datetime.now()
    return end_time - start_time


if __name__ == "__main__":
    size = 30000  # Adjust the size for desired complexity
    seed = 1  # Current seed for reproducibility

    time_cp = test_cupy_solve(size, seed)

    with open("results/performance_test_results.csv", "a", newline='') as csvfile:
        fieldnames = ['Matrix Size', 'Seed', 'Library', 'Time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header only if the file is empty
        if csvfile.tell() == 0:
            writer.writeheader()

        writer.writerow({'Matrix Size': size, 'Seed': seed, 'Library': 'CuPy', 'Time': time_cp})

    print(f"CuPy solve time: {time_cp}")
