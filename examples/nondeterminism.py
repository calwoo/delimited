"""
Replay-based non-determinism in Python.
"""

##################
# 2-choice version

# State-bit to record which branch to take during handling.
first_time = False

def choose2(x1, x2):
    if first_time:
        return x1
    else:
        return x2

def with_nondeterminism2(fn):
    """
    Handler for choice-- returns a reified list of
    all choices.
    """
    results = []
    # Needs to manipulate global state
    global first_time
    # Take first branch first
    first_time = True
    results.append(fn())
    # Take second branch
    first_time = False
    results.append(fn())
    return results

################################
# (>2)-choice version, used once

# Global state will be a pair, indicating current id of branch,
# and total number of branches
state = (None, None)

def start_idx(xs):
    return (0, len(xs))

def next_idx(k, length):
    if k + 1 == length:
        return (None, None)
    else:
        return (k + 1, length)

def get(xs, k, length):
    return xs[k]

def chooseM(xs):
    global state
    if len(xs) == 0:
        raise ValueError("it's the end")
    else:
        # Grab a value based on the current global state
        if state[0] is None:
            state = start_idx(xs)
            return get(xs, *state)
        else:
            return get(xs, *state)
    
def with_nondeterminismM(fn):
    """
    Handler for choice-- returns a reified list of
    all choices.
    """
    global state
    try:
        # Run the function with current state
        results = [fn()]
        if state[0] is None:
            return results
        else:
            if next_idx(*state)[0] is None:
                return results
            else:
                state = next_idx(*state)
                return results + with_nondeterminismM(fn)
    except ValueError:
        return []

######################
# General replay-based
# nondeterminism

# Global state will be in the form of a past and future stack.
# The past stack contains choices already made.
# The future stack contains known choices to be made.
past   = []
future = []

# The next path to choose is a modification of the current path through
# the final leaf.
def next_path(xs):
    if len(xs) == 0:
        return []
    else:
        i = xs[0]
        if next_idx(*i)[0] is None:
            return next_path(xs[1:])
        else:
            return [next_idx(*i)] + xs[1:]

"""
How is this supposed to work? When the execution of the handler reaches
a call to choose, it reads the choice from the future stack, and pushes
the remainder to the past.

If the future is unknown, then it means we have reached a choose statement
for the first time, at which we pick the first choice and record it in the
past stack.
"""
def choose(xs):
    global past, future
    if len(xs) == 0:
        raise ValueError("it's the end")
    else:
        if len(future) == 0:
            # If there is no future, start a new path index and
            # push it into the past.
            i = start_idx(xs)
            past.insert(0, i)
            return get(xs, *i)
        else:
            # Otherwise, read the instruction from the future stack
            # and execute, pushing back into the past.
            i = future.pop(0)
            past.insert(0, i)
            return get(xs, *i)

def with_nondeterminism(fn):
    global past, future
    try:
        results = [fn()]
        next_future = list(reversed(next_path(past)))
        # Reset past/future stacks
        past   = []
        future = next_future
        if len(future) == 0:
            return results
        else:
            return results + with_nondeterminism(fn)
    except ValueError:
        return []

####################
# Tests and examples

if __name__ == "__main__":
    results = with_nondeterminism2(lambda : 3 * choose2(5,6))
    print(results)

    results = with_nondeterminismM(lambda : 2 * chooseM([]))
    print(results)

    results = with_nondeterminismM(lambda : 2 * chooseM([1,2,3]))
    print(results)

    def test_fn():
        if choose([True, False]):
            return choose([1,2])
        else:
            return choose([3,4])

    results = with_nondeterminism(test_fn)
    print(results)

    def test_fn2():
        return 2 + choose([1,2,3]) * choose([1,10,100])

    results = with_nondeterminism(test_fn2)
    print(results)