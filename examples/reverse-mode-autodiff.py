import numpy as np
from delim.cont import *

class Dual:
    """
    Implementation of a dual element, i.e. an element of the
    Zariski tangent space.
    """
    def __init__(self, value, cont, grad=0):
        self.value = value
        self.grad  = grad
        self.C     = cont

    def __repr__(self):
        return "<value: {}> ~ <grad: {}>".format(self.value, self.grad)

    def add(self, other):
        """
        Addition with a continuation.
        """
        def add_fn(k):
            # Here, k is the continuation.
            # first, we perform the forward-pass
            y = Dual(self.value + other.value, self.C, 0.0)
            # perform the continuation
            k(y)
            # when it returns, update gradients in backward-pass
            self.grad += y.grad
            other.grad += y.grad
        return self.C.shift(lambda k: add_fn(k))

    def mul(self, other):
        """
        Multiplication with a continuation.
        """
        def mul_fn(k):
            # Here, k is the continuation.
            # first, we perform the forward-pass
            y = Dual(self.value * other.value, self.C, 0.0)
            # perform the continuation
            k(y)
            # when it returns, update gradients in backward-pass
            self.grad += other.value * y.grad
            other.grad += self.value * y.grad
        return self.C.shift(lambda k: mul_fn(k))

    def set_grad(self, new_grad):
        self.grad = new_grad

def grad(cont, fn):
    def grad_fn(x):
        z = Dual(x, cont, 0.0)
        def g():
            res = fn(z)
            res.set_grad(1.0)
        cont.reset(g)
        return z.grad
    return grad_fn

def eval(fn, x):
    val = Dual(x, grad=0)
    fn(lambda r: print(r))(val)

if __name__ == "__main__":
    C = Cont()
    # EXAMPLE 1:
    # fn(x) = 2x + x = 3x
    fn = lambda x: Dual(2.0, C).mul(x).add(x)
    # dfn(x) = 3
    dfn = grad(C, fn)
    
    print(dfn(500.0)) # => 3.0

    # EXAMPLE 2:
    # fn(x) = 2x + x^3
    # fn = lambda x: (x.mul(x.mul(x))).add(Dual(2.0, C).mul(x))
    fn = lambda x: x.mul(x).mul(x).add(x).add(x)
    # dfn(x) = 2 + 3x^2
    dfn = grad(C, fn)

    print(dfn(2.0)) # => 14.0
    