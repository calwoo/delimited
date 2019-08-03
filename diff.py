import numpy as np
from delim.cont import Cont

class Dual:
    """
    Implementation of a dual element, i.e. an element of the
    Zariski tangent space.
    """
    def __init__(self, value, grad=0):
        self.value = value
        self.grad = grad

    