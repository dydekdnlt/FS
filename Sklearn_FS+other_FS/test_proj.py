import numpy as np

u = np.array([2, 5, 8])

n = np.array([1, 1, 7])

n_norm = np.sqrt(sum(n**2))
print(n_norm)
proj_of_u_on_n = (np.dot(u, n)/n_norm**2)*n
print(proj_of_u_on_n)
print(u - proj_of_u_on_n)