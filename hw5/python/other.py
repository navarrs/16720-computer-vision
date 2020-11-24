import matplotlib.pyplot as plt
import numpy as np
from nn import *
from util import *

def softmax(x, c=0.0):
  s = np.exp(x + c)
  print(f"Num: {s}")
  S = np.sum(s)
  return s / S

x = np.array([-10, -2, -1, 0, 1, 2, 10])
print(softmax(x))

x = np.arange(-10, 11, 0.1)
dsig = sigmoid_deriv(sigmoid(x))

# plt.title('Derivative of Sigmoid')
# plt.show()
# plt.close()

def dtanh(x):
  return 1 - np.tanh(x)**2

dtan = dtanh(x)
plt.plot(x, dsig)
plt.plot(x, dtan, 'r')
plt.title('Derivative of tanh(x)')
plt.show()
plt.close()

