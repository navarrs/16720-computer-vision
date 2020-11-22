import numpy as np


def softmax(x, c=0.0):
  s = np.exp(x + c)
  print(f"Num: {s}")
  S = np.sum(s)
  return s / S

x = np.array([-10, -2, -1, 0, 1, 2, 10])
print(softmax(x))

print(softmax(x, -np.max(x)))