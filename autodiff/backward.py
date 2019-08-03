import numpy as np

class Dual:
    """
    Implementation of a dual element, i.e. an element of the
    Zariski tangent space.
    """
    def __init__(self, value, grad=0):
        self.value = value
        self.grad  = grad

    def __repr__(self):
        return "<value: {}> ~ <grad: {}>".format(self.value, self.grad)

    def add(self, other):
        """
        Addition with a continuation.
        """
        def add_fn(k):
            # Here, k is the continuation.
            # first, we perform the forward-pass
            y = Dual(self.value + other.value, 0.0)
            # perform the continuation
            k(y)
            # when it returns, update gradients in backward-pass
            self.grad += y.grad
            other.grad += y.grad
        return add_fn

    def mul(self, other):
        """
        Multiplication with a continuation.
        """
        def mul_fn(k):
            # Here, k is the continuation.
            # first, we perform the forward-pass
            y = Dual(self.value * other.value, 0.0)
            # perform the continuation
            k(y)
            # when it returns, update gradients in backward-pass
            self.grad += other.value * y.grad
            other.grad += self.value * y.grad
        return mul_fn

    def set_grad(self, new_grad):
        self.grad = new_grad

def grad(fn):
    def grad_fn(x):
        z = Dual(x, 0.0)
        fn(lambda r: r.set_grad(1.0))(z)
        return z.grad
    return grad_fn

def eval(fn, x):
    val = Dual(x, grad=0)
    fn(lambda r: print(r))(val)

if __name__ == "__main__":
    # EXAMPLE 1:
    # fn(x) = 2x + x = 3x
    fn = lambda k: (lambda x: Dual(2.0).mul(x)(lambda y: y.add(x)(lambda z: k(z))))
    # dfn(x) = 3
    dfn = grad(fn)
    
    print(dfn(500.0)) # => 3.0

    # EXAMPLE 2:
    # fn(x) = 2x + x^3
    fn = lambda k: (lambda x: Dual(2.0).mul(x)(
        lambda y1: x.mul(x)(
            lambda y2: y2.mul(x)(
                lambda y3: y1.add(y3)(
                    lambda z: k(z))))))
    # dfn(x) = 2 + 3x^2
    dfn = grad(fn)

    print(dfn(2.0)) # => 14.0
