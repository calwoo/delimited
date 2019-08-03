def fn1(k):
    print("enter fn1")
    k()
    print("exit fn1")

def fn2(k):
    print("enter fn2")
    k()
    print("exit fn2")

def fn3():
    print("fn3!")

fn1(lambda: fn2(lambda: fn3()))