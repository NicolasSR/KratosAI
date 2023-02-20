import numpy as np


v=np.random.rand(1,52)
print(v)
norm = np.linalg.norm(v)
v=v/norm

print(v)