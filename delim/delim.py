"""
Delimited continuations in Python.
"""

class Done(Exception):
    def __init__(self, value):
        self.value = value

# State is represented by a past, future stack, along
# with a stack to handle nested calls.
past = []
future = []
nest = []
cur_expr = None

# Key: Replay a computation with a known future. This is the
# "thermometer".
def thermometer(fn, fn_future):
    global past, future, nest, cur_expr
    # Push state of current reset block into nest stack
    nest.append((cur_expr, past.copy(), future.copy()))
    # Set up the thermometer
    past = []
    future = fn_future
    cur_expr = fn
    # Run the computation
    def run():
        try:
            return fn()
        except Done as e:
            return e.value
    result = run()
    # Undo the nesting
    try:
        # Set the thermometer state for recursive return
        prev_expr, prev_past, prev_future = nest.pop()
        cur_expr = prev_expr
        past = prev_past
        future = prev_future
        return result
    except IndexError:
        raise ValueError
    
def reset(fn):
    return thermometer(fn, [])

def shift(fn):
    global past, future, cur_expr
    # The thermometer (which is the future) contains the values of all
    # effectful computations that have perspired until this shift block.
    # If the next value in the future stack is a value, that means
    # that this is a replay of this frame, and not a new entry to this
    # shift block.
    case = None
    if len(future) == 0:
        case = 1
    else:
        val = future.pop(0)
        if val is None:
            case = 1
        else:
            case = 2
    # Case 1
    if case == 1:
        # During the replay, we'll replay the entire computation with the
        # state set to the value called in shift, so that on the next-pass
        # the other if condition will ignore this shift block.
        new_future = list(reversed(past))
        our_expr = cur_expr
        def k(v):
            return thermometer(our_expr, new_future + [v])
        past.append(None)
        # Recursively call the replay
        result = fn(k)
        # When we hit a result, create an exception to abort the computation in
        # the reset block so that we don't perform the further computation outside
        # of the shift blocks.
        raise Done(result)
    # Case 2
    elif case == 2:
        past.append(val)
        return val

if __name__ == "__main__":
    ex1 = reset(lambda: 2 * shift(lambda k: 1 + k(5)))
    print(ex1) # => 11

    ex2 = reset(lambda: 1 + shift(lambda k: k(1) * k(2) * k(3)))
    print(ex2) # => 24

    ex3 = 1 + reset(lambda: 2 + shift(lambda k:
        3 * shift(lambda l: l(k(10)))))
    print(ex3) # => 37