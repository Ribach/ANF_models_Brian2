import thorns as th
import numpy as np

# =============================================================================
# Test map function
# =============================================================================
def multiply(a, b):
    """Multiply two numbers."""

    # The output should be wrapped in a dict
    y = {'y': a * b}

    return y


# Keys in the input dict MUST correspond to the key word
# arguments of the function (e.g. multiply)
xs = {
    'a': np.arange(150),
    'b': np.arange(150)
}

# Uncomment `backend` for parallel execution
ys = th.util.map(func = multiply,
                 space = xs,
                 backend='multiprocessing',
                 cache = 'no'
)
