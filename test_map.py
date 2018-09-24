import thorns as th
import numpy as np

# =============================================================================
# Test map function
# =============================================================================
def multiply(a, b, c):
    """Multiply two numbers."""

    # The output should be wrapped in a dict
    y = {'y': a * b + c}

    return y


# Keys in the input dict MUST correspond to the key word
# arguments of the function (e.g. multiply)
xs = {
    'a': np.arange(100),
    'b': np.arange(100),
}


# Uncomment `backend` for parallel execution
ys = th.util.map(func = multiply,
                 space = xs,
                 backend='serial',
                 cache = 'no',
                 kwargs = {'c' : 3}
)
