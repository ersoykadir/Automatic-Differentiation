import numpy as np
import time
from copy import deepcopy


class Number:
    def __init__(self, value, grad):
        self.value = value
        self.grad = grad
    
    def __str__(self):
        return f"(v: {self.value}, g: {self.grad})"

    def __repr__(self):
        return f"(v: {self.value}, g: {self.grad})"
    
    def __add__(self, other):
        value = self.value + other.value
        grad = self.grad + other.grad
        return Number(value, grad)

    def __mul__(self, other):
        value = self.value * other.value
        grad = self.grad * other.value + self.value * other.grad
        return Number(value, grad)

    def __sub__(self, other):
        value = self.value - other.value
        grad = self.grad - other.grad
        return Number(value, grad)

    def __truediv__(self, other):
        value = self.value / other.value
        grad = (self.grad * other.value - self.value * other.grad) / (other.value ** 2)
        return Number(value, grad)
    
    def __pow__(self, k):
        if k == 0:
            return Number(1.0, 0.0)

        t = deepcopy(self)
        for i in range(k-1):
            t = self * t
        return t


def relu(x):
    if x.value > 0:
        return x
    else:
        return Number(0.0, 0.0)


def babylonian_sqrt(x):
    t = Number(1.0, 0.0)
    for i in range(10):
        t = (t+x/t) / Number(2.0, 0.0)
    return t

def sin(x):
    pi = 3.141592653589793
    if x.value > 2*pi:
        time = x.value // (2*pi)
        x.value = x.value - 2*pi*time
    t = Number(0.0, 0.0)
    minus_x_2 = Number(0.0, 0.0) - x**2
    numerator = deepcopy(x)
    denominator = Number(1.0, 0.0)
    for i in range(1, 20):
        t = t + numerator / denominator
        numerator = numerator * minus_x_2
        denominator = denominator * Number(2*i*(2*i+1), 0.0)
    return t

def cos(x):
    pi = 3.141592653589793
    if x.value > 2*pi:
        time = x.value // (2*pi)
        x.value = x.value - 2*pi*time
    t = Number(0.0, 0.0)
    minus_x_2 = Number(0.0, 0.0) - x**2
    numerator = Number(1.0, 0.0)
    denominator = Number(1.0, 0.0)
    for i in range(1, 20):
        t = t + numerator/denominator
        numerator = numerator * minus_x_2
        denominator = denominator * Number(2*i*(2*i-1), 0.0)
    return t


r = np.random.randn()
# auto-differentiation result
print(sin(Number(r, 1.0)).grad)
# calculate the gradient manually
print(np.cos(r))

r = np.random.randn()
# auto-differentiation result
print(cos(Number(r, 1.0)).grad)
# calculate the gradient manually
print(-np.sin(r))

# here comes the magic...
x = Number(5, 0)
w = Number(12, 1)
b = Number(7, 0)
z = Number(8, 0)
y = cos(sin((w**2)*x+b)+w**2)
# auto-differentiation result
print(y.grad)
# calculate the gradient manually
print(-np.sin(np.sin(144*5+7)+144)*(np.cos(144*5+7)*2*12*5 + 2*12))
