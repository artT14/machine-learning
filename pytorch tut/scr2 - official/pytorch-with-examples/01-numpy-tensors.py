# LINK: https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#warm-up-numpy
import numpy as np
import math
"""
use numpy to fit a third order polynomial to sine
function by manually implementing the forward and
backward passes through the network using numpy operations
"""
# CREATE RANDOM INPUT & OUTPUT DATA
x = np.linspace(-math.pi, math.pi,2000)
y = np.sin(x)
print(x)
print(len(x))
print(y)
print(len(y))

# RANDOMLY INITIALIZE WEIGHTS
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()
print(a,b,c,d)

learning_rate = 1e-6
for t in range(2000):
    # FORWARD PASS: COMPUTE PREDICTED y
    # y = a + bx +cx^2 + dx^3
    y_pred = a + b*x + c*x**2 + d*x**3
    
    # COMPUTE & PRINT LOSS
    loss = np.square(y_pred - y).sum()
    if t % 100 == 99:
        print(t,loss)
    
    # BACKPROP TO COMPUTE GRADIENTS OF a, b, c, d w/ respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()
    
    # UPDATE WEIGHTS
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')