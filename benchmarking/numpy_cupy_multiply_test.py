from datetime import datetime

import numpy as np
import cupy as cp

np.random.seed(0)
cp.random.seed(0)

# A_np = np.random.rand(25000, 25000)
# B_np = np.random.rand(25000, 25000)
#
# start = datetime.now()
# C_np = np.dot(A_np, B_np)
# print("Time taken (NumPy):", datetime.now() - start)

A_cp = cp.random.rand(30000, 30000)
B_cp = cp.random.rand(30000, 30000)

start = datetime.now()
C_cp = cp.dot(A_cp, B_cp)
cp.cuda.Stream.null.synchronize()
print("Time taken (CuPy):", datetime.now() - start)
