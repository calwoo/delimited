import numpy as np
from delim.cont import Cont

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

    def __radd__(self, other):
        """
        Addition with a continuation.
        """
        # Cast to Zariski tangent space
        if not isinstance(other, Dual):
            other = self.cast(other)
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

    def __add__(self, other):
        """
        Addition with a continuation.
        """
        # Cast to Zariski tangent space
        if not isinstance(other, Dual):
            other = self.cast(other)
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

    def __rmul__(self, other):
        """
        Multiplication with a continuation.
        """
        # Cast to Zariski tangent space
        if not isinstance(other, Dual):
            other = self.cast(other)
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

    def __mul__(self, other):
        """
        Multiplication with a continuation.
        """
        # Cast to Zariski tangent space
        if not isinstance(other, Dual):
            other = self.cast(other)
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

    def __pow__(self, p):
        # IGNORE LOG CASE
        if p != -1:
            def pow_fn(k):
                y = Dual(self.value**p, self.C, 0.0)
                k(y)
                self.grad += p * self.value**(p-1) * y.grad
            return self.C.shift(lambda k: pow_fn(k))

    def cast(self, num):
        # Cast as a dual element
        return Dual(num, self.C, grad=0)

    def set_grad(self, new_grad):
        self.grad = new_grad

class Autodifferentiator:
    """
    Implementation of a contained autodifferentiator, with
    an enclosed set of delimited control operators 
    (shift/reset), along with implementations of dual elements
    of a Zariski tangent space for purposes of reverse-mode
    automatic differentiation.
    """
    def __init__(self):
        self.C = Cont()

    def grad(self, fn):
        def grad_fn(x):
            z = Dual(x, self.C, 0.0)
            def g():
                res = fn(z)
                res.set_grad(1.0)
            self.C.reset(g)
            return z.grad
        return grad_fn


if __name__ == "__main__":
    auto = Autodifferentiator()
    
    fn = lambda x: x**3 + 3*x
    dfn = auto.grad(fn)
    print(dfn(4.0))