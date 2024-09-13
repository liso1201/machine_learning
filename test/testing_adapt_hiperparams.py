import sys
sys.path.append('.')

import numpy as np
import pandas as pd

from collections import deque
from tune import adapt_hiperparams

test_cases = {
    "Scalar int": 5,
    "Scalar float": 3.14,
    "Scalar complex": 2 + 3j,
    "Scalar string": "hello",
    "List of ints": [1, 2, 3],
    "Tuple of floats": (4.5, 5.5),
    "Set of strings": {"a", "b", "c"},
    "Ndarray": np.array([1, 2, 3]),
    "Series": pd.Series([4, 5, 6]),
    "Generator": (x for x in range(3)),
    "Deque": deque([7, 8, 9]),
    "Empty list": [],
    "Empty tuple": (),
    "Empty set": set(),
    "Empty generator": (x for x in []),
    "String with newline": "hello\nworld",
    "String with spaces": "   spaced  string   "
}

print(adapt_hiperparams(test_cases))


