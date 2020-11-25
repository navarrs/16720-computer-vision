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

def dtanh(x):
  return 1 - np.tanh(x)**2

plt.plot(x, sigmoid_deriv(sigmoid(x)), label='dsigmoid')
plt.plot(x, dtanh(x), 'r', label='dtanh')
plt.title('Derivative of activation functions')
plt.legend(loc='upper left', borderaxespad=0.)
plt.savefig("../out/q1/derivs.png")
plt.show()
plt.close()
