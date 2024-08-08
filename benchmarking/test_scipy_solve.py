import numpy as np
import scipy as sp
from datetime import datetime
import csv


def generate_large_system_scipy(size, seed):
    sp.random.seed(seed)
    A = sp.random.rand(size, size)
    B = sp.random.rand(size)
    return A, B


def test_scipy_solve(size, seed):
    A, B = generate_large_system_scipy(size, seed)
    start_time = datetime.now()
    X = sp.linalg.solve(A, B)
    end_time = datetime.now()
    return end_time - start_time


if __name__ == "__main__":
    size = 30000  # Adjust the size for desired complexity
    seed = 1  # Current seed for reproducibility

    time_sp = test_scipy_solve(size, seed)

    with open("results/performance_test_results.csv", "a", newline='') as csvfile:
        fieldnames = ['Matrix Size', 'Seed', 'Library', 'Time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header only if the file is empty
        if csvfile.tell() == 0:
            writer.writeheader()

        writer.writerow({'Matrix Size': size, 'Seed': seed, 'Library': 'SciPy', 'Time': time_sp})

    print(f"SciPy solve time: {time_sp}")
