import numpy as np
from datetime import datetime
import csv


def generate_large_system_numpy(size, seed):
    np.random.seed(seed)
    A = np.random.rand(size, size)
    B = np.random.rand(size)
    return A, B


def test_numpy_solve(size, seed):
    A, B = generate_large_system_numpy(size, seed)
    start_time = datetime.now()
    X = np.linalg.solve(A, B)
    end_time = datetime.now()
    return end_time - start_time


if __name__ == "__main__":
    size = 30000  # Adjust the size for desired complexity
    seed = 1  # Current seed for reproducibility

    time_np = test_numpy_solve(size, seed)

    with open("results/performance_test_results.csv", "a", newline='') as csvfile:
        fieldnames = ['Matrix Size', 'Seed', 'Library', 'Time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header only if the file is empty
        if csvfile.tell() == 0:
            writer.writeheader()

        writer.writerow({'Matrix Size': size, 'Seed': seed, 'Library': 'NumPy', 'Time': time_np})

    print(f"NumPy solve time: {time_np}")
