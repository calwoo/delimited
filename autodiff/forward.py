"""
Forward-mode autodifferentiation could just be
implemented using operator overloading.
"""

class Tangent:
    """
    Implementation of an element of the Zariski
    tangent space.
    """
    def __init__(self, value, grad=0):
        self.value = value
        self.grad  = grad

    def __repr__(self):
        return "<value: {}> ~ <grad: {}>".format(self.value, self.grad)

    def __add__(self, other):
        v = self.value + other.value
        g = self.grad + other.grad
        return Tangent(v, g)

    def __mul__(self, other):
        v = self.value * other.value
        g = self.grad * other.value + self.value * other.grad
        return Tangent(v, g)
    
def num(x):
    # Cast number as Tangent
    return Tangent(x, 0.0)

def grad(fn):
    def grad_fn(x):
        y = fn(Tangent(x, 1.0))
        return y.grad
    return grad_fn


if __name__ == "__main__":
    a = Tangent(5, 0)
    b = Tangent(6, 1)

    x = a*a*b
    print("a = ", a)
    print("b = ", b)
    print("x = ", x)

    def fn(x):
        return num(2)*x + x*x*x

    dfn = grad(fn)
    print(dfn(1))