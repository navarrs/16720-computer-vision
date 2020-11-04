'''
Q5.2:
'''
from submission import (
    rodrigues,
    invRodrigues
)
import numpy as np

r = np.array([1.57, 0, 0]).T
R = rodrigues(r)
r_ = invRodrigues(R)
print(f"r: {r}\nR: {R}\nr_: {r_}")