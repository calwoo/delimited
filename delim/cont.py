"""
Packing up delimited continuations into a single
class for modularization.
"""

class Done(Exception):
    def __init__(self, value):
        self.value = value

class Enter:
    @property
    def __init__(self):
        pass

class Cont:
    """
    Implementation class for delimited continuations via
    the shift/reset interface of Danvy-Filinski.
    """
    def __init__(self):
        """
        Initialize global state necessary for delimited continuations.
        Implementation follows the functional pearl
            "Capturing the past by replaying the future."
        """
        self.past = []
        self.future = []
        self.nest = []
        self.cur_expr = None

    @property
    def reset(self):
        self.past = []
        self.future = []
        self.nest = []
        self.cur_expr = None

    # Key: Replay a computation with a known future. This is the
    # "thermometer".
    def _thermometer(self, fn, fn_future):
        # Push state of current reset block into nest stack
        self.nest.append((self.cur_expr, self.past.copy(), self.future.copy()))
        # Set up the thermometer
        self.past = []
        self.future = fn_future
        self.cur_expr = fn
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
            prev_expr, prev_past, prev_future = self.nest.pop()
            self.cur_expr = prev_expr
            self.past = prev_past
            self.future = prev_future
            return result
        except IndexError:
            raise ValueError

    def reset(self, fn):
        return self._thermometer(fn, [])

    def shift(self, fn):
        # The thermometer (which is the future) contains the values of all
        # effectful computations that have perspired until this shift block.
        # If the next value in the future stack is a value, that means
        # that this is a replay of this frame, and not a new entry to this
        # shift block.
        case = None
        if len(self.future) == 0:
            case = 1
        else:
            val = self.future.pop(0)
            if val is None:
                case = 1
            else:
                case = 2
        # Case 1
        if case == 1:
            # During the replay, we'll replay the entire computation with the
            # state set to the value called in shift, so that on the next-pass
            # the other if condition will ignore this shift block.
            new_future = list(reversed(self.past))
            our_expr = self.cur_expr
            def k(v):
                return self._thermometer(our_expr, new_future + [v])
            self.past.append(None)
            # Recursively call the replay
            result = fn(k)
            # When we hit a result, create an exception to abort the computation in
            # the reset block so that we don't perform the further computation outside
            # of the shift blocks.
            raise Done(result)
        # Case 2
        elif case == 2:
            self.past.append(val)
            return val

if __name__ == "__main__":
    C = Cont()

    ex1 = C.reset(lambda: 2 * C.shift(lambda k: 1 + k(5)))
    print(ex1) # => 11

    ex2 = C.reset(lambda: 1 + C.shift(lambda k: k(1) * k(2) * k(3)))
    print(ex2) # => 24

    ex3 = 1 + C.reset(lambda: 2 + C.shift(lambda k:
        3 * C.shift(lambda l: l(k(10)))))
    print(ex3) # => 37