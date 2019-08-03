"""
Baby thermometer continuations in Python.
"""

class Done(Exception):
    def __init__(self, value):
        self.value = value

state = None
cur_expr = None

def reset(fn):
    global state, cur_expr
    cur_expr = fn
    state = None
    try:
        fn()
    except Done as e:
        return e.value

def shift(fn):
    global state
    # If state is not None, that means that we are performing a replay
    # where a value has been passed into the continuation. Then here, shift
    # will act as if the entire block is that value, for the purpose of
    # performing the delimited continuation.
    if state is not None:
        return state
    else:
        # During the replay, we'll replay the entire computation with the
        # state set to the value called in shift, so that on the next-pass
        # the other if condition will ignore this shift block.
        def k(x):
            global state
            state = x
            return cur_expr()
        # Recursively call the replay
        result = fn(k)
        # When we hit a result, create an exception to abort the computation in
        # the reset block so that we don't perform the further computation outside
        # of the shift blocks.
        raise Done(result)

if __name__ == "__main__":
    ex1 = reset(lambda: 2 * shift(lambda k: 1 + k(5)))
    print(ex1) # => 11

    ex2 = reset(lambda: 1 + shift(lambda k: k(1) * k(2) * k(3)))
    print(ex2) # => 24