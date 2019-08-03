from delim.cont import *

C = Cont()

def choose(x, y):
    return C.shift(lambda k: [k(x)] + [k(y)])

def values(fn):
    return C.reset(fn)

if __name__ == "__main__":
    ex1 = values(lambda: choose(0,1))
    print(ex1)

    ex2 = values(lambda: choose(1, choose(2,3)))
    print(ex2)